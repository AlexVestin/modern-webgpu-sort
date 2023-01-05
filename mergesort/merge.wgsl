R"(
  struct Parameters {
    count: u32;
    nt: u32;
    vt: u32;
    num_wg: u32;
    num_partition: u32;
  };

  struct Data { data: array<u32>; };
  struct Counter { data: u32; };

  @binding(0) @group(0) var<storage, read> keys: Data;
  @binding(1) @group(0) var<storage, write> keys_out: Data;
  @binding(2) @group(0) var<uniform> params: Parameters;
  @binding(3) @group(0) var<storage, write> partitions: Data;
  @binding(4) @group(0) var<storage, read> counter: Counter;

  // nt * vt (128 * 15)
  var<workgroup> shared: array<u32, 2048>;
  var<private> local_keys: array<u32, 16>;

  fn reg_to_shared_thread(tid: u32) {
  for(var i = 0u; i < 15u; i = i + 1u) {
      shared[15u*tid+i] = local_keys[i];
    }
    workgroupBarrier();
  }

  fn shared_to_reg_strided(tid: u32) {
    for(var i = 0u; i < 15u; i = i + 1u) {
      local_keys[i] = shared[128u * i + tid];
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

  fn compute_mergesort_frame(partition: i32, coop: i32, spacing: i32) -> vec4<i32> {
    let size = spacing * (coop / 2);
    let start = ~(coop - 1) & partition;
    let a_begin = spacing * start;
    let b_begin = spacing * start + size;
    return vec4<i32>(
      a_begin,
      a_begin + size,
      b_begin,
      b_begin + size
    );
  }

  fn partition(range: vec4<i32>, mp0: i32, diag: i32) -> vec4<i32> {
    return vec4<i32>(range.x + mp0, range.y, range.z + diag - mp0, range.w);
  }

  fn compute_mergesort_range(count: i32, partition: i32, coop: i32, spacing: i32) -> vec4<i32> {
    let frame = compute_mergesort_frame(partition, coop, spacing);
    return vec4<i32>(
      frame.x,
      min(count, frame.y),
      min(count, frame.z),
      min(count, frame.w)
    );
  }

  fn compute_mergesort_range_2(count: i32, partition: i32, coop: i32, spacing: i32, mp0: i32, mp1: i32) -> vec4<i32> {
    var range = compute_mergesort_range(count, partition, coop, spacing);
    let diag = spacing * partition - range.x;

    if(coop - 1 != ((coop - 1) & partition)) {
      range.y = range.x + mp1;
      range.w = min(count, range.z + diag + spacing - mp1);
    }

    range.x = range.x + mp0;
    range.z = min(count, range.z + diag - mp0);
    return range;
  }

  fn merge_path_2(a_keys: u32, a_count: u32, b_keys: u32, b_count: u32, diag: u32) -> u32 {
    var begin = u32(max(0, i32(diag - b_count)));
    var end   = min(diag, a_count);

    loop {
      if(begin >= end) {
        break;
      }
      let mid = (begin + end) / 2u;
      let a_key = shared[a_keys + mid];
      let b_key = shared[b_keys + diag - 1u - mid];

      if (a_key <= b_key) {
        begin = mid + 1u;
      } else {
        end = mid;
      }
    }

    return begin;
  }

  fn get_tile(cta: u32, nv: u32, count: u32) -> vec2<u32> {
    return vec2<u32>(nv * cta, min(count, nv * (cta + 1u)));
  }

  fn to_local(range: vec4<i32>) -> vec4<i32> {
    return vec4<i32>(
      0, 
      range.y-range.x, 
      range.y-range.x, 
      (range.y-range.x) + (range.w-range.z)
    );
  }

  fn merge_predicate(range: vec4<i32>, a_key: u32, b_key: u32) -> bool {
    if(range.x >= range.y) { return false; }
    if(range.z >= range.w) { return true; }
    return a_key < b_key;
  }

  fn serial_merge(range_: vec4<i32>) {
    var crange = range_;
    var a_key = shared[crange.x];
    var b_key = shared[crange.z];
    
    for(var i = 0u; i < 15u; i = i + 1u) {
      let p = merge_predicate(crange, a_key, b_key);
      var index = crange.z;
      if (p) {
        index = crange.x;
      }
      
      if (p) {
        local_keys[i] = a_key;
      } else {
        local_keys[i] = b_key;
      }

      let c_key = shared[u32(index) + 1u];
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

  fn merge_path(range: vec4<i32>, diag: u32) -> u32 {
    return merge_path_2(
      u32(range.x), 
      u32(range.y - range.x), 
      u32(range.z), 
      u32(range.w - range.z),
      diag 
    );
  }

  fn reg_to_shared_strided(tid: u32) {
    for(var i = 0u; i < 15u; i = i + 1u) {
      shared[128u * i + tid] = local_keys[i];
    }
    workgroupBarrier();
  }

  fn load_two_streams_reg(a: u32, a_count: u32, b: u32, b_count: u32, tid: u32) {
    let bb = b - a_count;
    let count = a_count + b_count;
    if (count >= 128u * 15u) {
      // No checking
      // TODO: break out into own function
      for (var i = 0u; i < 15u; i = i + 1u) {
        if(128u * i + tid >= a_count) {
          local_keys[i] = keys.data[bb + 128u * i + tid];
        } else {
          local_keys[i] = keys.data[a + 128u * i + tid];
        }
      }
    } else {
      for (var i = 0u; i < 15u; i = i + 1u) {
        let j = 128u * i + tid;
        if(j < count) {
          if(128u * i + tid >= a_count) {
            local_keys[i] = keys.data[bb + 128u * i + tid];
          } else {
            local_keys[i] = keys.data[a + 128u * i + tid];
          }
        }
      }
    }

    workgroupBarrier();
  }

  fn load_two_streams_shared(a_begin: u32, a_count: u32, b_begin: u32, b_count: u32, tid: u32) {
    // Load into register then make an unconditional strided store into memory.
    load_two_streams_reg(a_begin, a_count, b_begin, b_count, tid);
    reg_to_shared_strided(tid);
  }
  
  fn cta_merge_from_mem(range_mem: vec4<i32>, tid: u32, workgroup_id: vec3<u32>) { 
    load_two_streams_shared(
      u32(range_mem.x), 
      u32(range_mem.y - range_mem.x), 
      u32(range_mem.z), 
      u32(range_mem.w - range_mem.z),
      tid
    );

    let range_local = to_local(range_mem);
    let diag = 15u * tid;
    let mp = merge_path(range_local, diag);

    let part = partition(range_local, i32(mp), i32(diag));

    serial_merge(part);
  }
 
  @stage(compute) @workgroup_size(128, 1, 1)
  fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(global_invocation_id) global_id: vec3<u32>
  ) {
    // SETUP
    let nv = 128u * 15u;   
    let pass = (counter.data / params.num_wg) - 1u;
    let coop = 2 << pass;
    // let coop = params.coop;

    let tile = get_tile(workgroup_id.x, nv, params.count);
    let range = compute_mergesort_range_2(
      i32(params.count), 
      i32(workgroup_id.x), 
      i32(coop), 
      i32(nv),
      i32(partitions.data[workgroup_id.x]),
      i32(partitions.data[workgroup_id.x + 1u])
    );

    cta_merge_from_mem(range, local_id.x, workgroup_id);
    reg_to_mem_thread(tile.x, local_id.x, tile.y - tile.x);
  }
)"