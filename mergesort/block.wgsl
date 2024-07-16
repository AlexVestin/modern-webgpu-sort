R"(
  
  struct Parameters {
    count: u32,
    nt: u32,
    vt: u32,
    coop: u32,
    num_partition: u32,
  };

  struct Data { data: array<u32> };
  // Needs to write since it may be used as both input and output
  @binding(0) @group(0) var<storage, read_write> keys: Data;
  @binding(1) @group(0) var<storage, read_write> keys_out: Data;
  @binding(2) @group(0) var<uniform> params: Parameters;

  // nt * vt (128 * 15)
  var<workgroup> shared_: array<u32, 2048>;
  var<private> local_keys: array<u32, 16>;

  fn mem_to_reg_strided(global_offset: u32, tid: u32, count: u32) {
    if (15u > 1u && count >= 128u * 15u) {
      for (var i = 0u; i < 15u; i = i + 1u) {
        local_keys[i] = keys.data[global_offset + 128u*i+tid];
      }
    } else {
      for (var i = 0u; i < 15u; i = i + 1u) {
        let j = 128u * i + tid;
        if(j < count) {
          local_keys[i] = keys.data[global_offset + 128u*i+tid];
        }
      }   
    }
  }

  fn reg_to_shared_strided(tid: u32) {
    for(var i = 0u; i < 15u; i = i + 1u) {
      shared_[128u * i + tid] = local_keys[i];
    }
  }

  fn shared_to_reg_thread(tid: u32) {
    for (var i = 0u; i < 15u; i = i +1u) {
      local_keys[i] = shared_[15u * tid + i];
    }
  }

  fn mem_to_reg_thread(global_offset: u32, tid: u32, count: u32) {
    mem_to_reg_strided(global_offset, tid, count);
    workgroupBarrier();
    reg_to_shared_strided(tid);
    workgroupBarrier();
    shared_to_reg_thread(tid);
    workgroupBarrier();
  }

  fn reg_to_shared_thread(tid: u32) {
    for(var i = 0u; i < 15u; i = i + 1u) {
      shared_[15u*tid+i] = local_keys[i];
    }
    workgroupBarrier();
  }

  fn shared_to_reg_strided(tid: u32) {
    for(var i = 0u; i < 15u; i = i + 1u) {
      local_keys[i] = shared_[128u * i + tid];
    }
    workgroupBarrier();
  }

  fn reg_to_mem_strided(global_offset: u32, tid: u32, count: u32) {
    if (15u > 1u && count >= 128u * 15u) {
      for (var i = 0u; i < 15u; i = i + 1u) {
        keys_out.data[global_offset + 128u*i+tid] = local_keys[i];
      }
    } else {
      for (var i = 0u; i < 15u; i = i + 1u) {
        let j = 128u * i + tid;
        if(j < count) {
          keys_out.data[global_offset + 128u*i+tid] = local_keys[i];
        }
      }   
    }
  }

  fn reg_to_mem_thread(global_offset: u32, tid: u32, count: u32) {
    reg_to_shared_thread(tid);
    workgroupBarrier();
    shared_to_reg_strided(tid);
    workgroupBarrier();
    reg_to_mem_strided(global_offset, tid, count);
  }

  fn out_of_range_flags(first: i32, vt: i32, count: i32) -> i32 {
    let out_of_range = min(vt, first + vt - count);
    var head_flags = 0;
    if (out_of_range > 0) {
      let mask = (1 << u32(vt)) - 1;
      head_flags = mask & ( ~mask >> u32(out_of_range));
    }
    return head_flags;
  }

  fn compare(i: u32, j: u32) -> bool {
    return local_keys[i] < local_keys[j];
  }

  fn swap(i: u32, j: u32) {
    // TODO offset i & j here
    let temp = local_keys[i];
    local_keys[i] = local_keys[j];  
    local_keys[j] = temp;
  }

  fn odd_even_sort(flags: i32, vt: u32) {
    for(var j = 0u; j < vt; j = j + 1u) {
      for(var i = 1u & j; i < vt - 1u; i = i + 2u) {
        if((0 == ((2 << i) & flags)) && compare(i + 1u, i)) {
          swap(i, i + 1u);
        }
      }
    }
  }

  fn compute_mergesort_frame(partition_: i32, coop: i32, spacing: i32) -> vec4<i32> {
    let size = spacing * (coop / 2);
    let start = ~(coop - 1) & partition_;
    let a_begin = spacing * start;
    let b_begin = spacing * start + size;
    return vec4<i32>(
      a_begin,
      a_begin + size,
      b_begin,
      b_begin + size
    );
  }

  fn partition_(range: vec4<i32>, mp0: i32, diag: i32) -> vec4<i32> {
    return vec4<i32>(range.x + mp0, range.y, range.z + diag - mp0, range.w);
  }

  fn compute_mergesort_range(count: i32, partition_: i32, coop: i32, spacing: i32) -> vec4<i32> {
    let frame = compute_mergesort_frame(partition_, coop, spacing);
    return vec4<i32>(
      frame.x,
      min(count, frame.y),
      min(count, frame.z),
      min(count, frame.w)
    );
  }

  fn get_tile(cta: u32, nv: u32, count: u32) -> vec2<u32> {
    return vec2<u32>(nv * cta, min(count, nv * (cta + 1u)));
  }

  fn s_log2(x: u32) -> u32 {
    if (x <= 1u) { return 0u; }

    var v = x;
    var c = 0u;
    loop { 
      v = v / 2u; 
      if(v == 0u) { break; } 
      c = c + 1u; 
    }
    return c;
  }

  fn merge_path_2(a_keys: u32, a_count: u32, b_keys: u32, b_count: u32, diag: u32) -> u32 {
    var begin = u32(max(0, i32(diag - b_count)));
    var end   = min(diag, a_count);

    loop {
      if(begin >= end) {
        break;
      }
      let mid = (begin + end) / 2u;
      let a_key = shared_[a_keys + mid];
      let b_key = shared_[b_keys + diag - 1u - mid];

      if (a_key <= b_key) {
        begin = mid + 1u;
      } else {
        end = mid;
      }
    }

    return begin;
  }

  fn merge_path(range: vec4<i32>, diag: u32) -> u32 {
    return merge_path_2(
      u32(range.x), 
      u32(range.y - range.x), 
      u32(range.z), 
      u32(range.w - range.z),
      diag 
    );
  }

  fn merge_predicate(range: vec4<i32>, a_key: u32, b_key: u32) -> bool {
    if(range.x >= range.y) { return false; }
    if(range.z >= range.w) { return true; }
    return a_key < b_key;
  }

  fn serial_merge(range_: vec4<i32>) {
    var crange = range_;
    var a_key = shared_[crange.x];
    var b_key = shared_[crange.z];
    var temp: array<u32, 15>;
    
    for(var i = 0u; i < 15u; i = i + 1u) {
      let p = merge_predicate(crange, a_key, b_key);
      var index = crange.z;
      if (p) {
        index = crange.x;
      }
      
      if (p) {
        local_keys[i] = a_key;
        //temp[i] = a_key;
      } else {
        // temp[i] = b_key;
        local_keys[i] = b_key;
      }

      let c_key = shared_[u32(index) + 1u];
      if (p) {
        a_key = c_key;
        crange.x = index + 1; 
      } else {
        b_key = c_key;
        crange.z = index + 1; 
      }
    }

    workgroupBarrier();

  }

  fn merge_pass(global_offset: u32, tid: u32, count: u32, pass_: u32) {
    let coop =  2 << pass_;
    let range = compute_mergesort_range(i32(count), i32(tid), i32(coop), i32(15u));
    let diag = 15u * tid - u32(range.x); 

    reg_to_shared_thread(tid);
    let mp = merge_path(range, diag);
    let part = partition_(range, i32(mp), i32(diag));

    shared_to_reg_thread(tid);

    serial_merge(part);
  }

  fn block_sort(global_offset: u32, tid: u32, count: u32, nt: u32, vt: u32) {
    if (count < 128u * vt) {
      let head_flags = out_of_range_flags(i32(vt * tid), i32(vt), i32(count));
      odd_even_sort(head_flags, vt);
    } else {
      odd_even_sort(0, vt);
    }

    let num_passes = s_log2(nt);
    // Merge threads starting with a pair until all values are merged.
    for (var pass_ = 0u; pass_ < num_passes; pass_++) {
      merge_pass(global_offset, tid, count, pass_);
    } 
  }

  @compute @workgroup_size(128, 1, 1)
  fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(global_invocation_id) global_id: vec3<u32>
  ) {
    // SETUP
    let nv = 128u * 15u;
    let tile = get_tile(workgroup_id.x, nv, params.count);
    let tile_count = tile.y - tile.x;

    // LOAD STUFF
    mem_to_reg_thread(tile.x, local_id.x, tile_count);

    // SORT
    block_sort(tile.x, local_id.x, tile_count, 128u, 15u);
    workgroupBarrier();

    // STORE STUFF
    reg_to_mem_thread(tile.x, local_id.x, tile_count);
  }
)"