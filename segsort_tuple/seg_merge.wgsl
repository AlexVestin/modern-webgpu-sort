R"(
  struct Parameters {
    count: u32,
    nt: u32,
    vt: u32,
    num_wg: u32,
    num_partitions: u32,
    num_segments: u32,
    num_ranges: u32,
    num_partition_ctas: u32,
    max_num_passes: u32
  };

  struct Data2 { data: array<vec2<u32>> };
  struct Data { data: array<u32> };
  struct MergeRanges { data: array<vec4<i32>> };
  struct Counter { data: u32 };

  @binding(0) @group(0) var<storage, read> keys_src: Data2;
  @binding(1) @group(0) var<storage, read_write> keys_dst: Data2;
  @binding(2) @group(0) var<uniform> params: Parameters;
  @binding(3) @group(0) var<storage, read> merge_list: MergeRanges;
  @binding(4) @group(0) var<storage, read> compressed_ranges: Data;
  @binding(5) @group(0) var<storage, read> pass_counter: Counter;

  var<workgroup> shared_: array<vec2<u32>, 1921>;
  var<private> local_keys: array<vec2<u32>, 15>;

  fn unpack_x(val: u32) -> i32 {
    return i32(val & 0xffffu); 
  }

  fn unpack_y(val: u32) -> i32 {
    return i32(val >> 16u);
  }

  fn comp(a_key: u32, b_key: u32) -> bool {
    let ay = unpack_y(a_key);
    let by = unpack_y(b_key);
    if (ay < by) {
      return true;
    }

    if (ay == by) {
      return unpack_x(a_key) < unpack_x(b_key);
    }

    return false;
  }

  fn load_two_streams_reg(a: u32, a_count: u32, b: u32, b_count: u32, tid: u32) {
    let bb = b - a_count;
    let count = a_count + b_count;
    if (count >= 128u * 15u) {

      for (var i = 0u; i < 15u; i = i + 1u) {
        let j = 128u * i + tid;
        if(j >= a_count) {
          local_keys[i] = keys_src.data[bb + j];
        } else {
          local_keys[i] = keys_src.data[a + j];
        }
      }
    } else {
      for (var i = 0u; i < 15u; i = i + 1u) {
        let j = 128u * i + tid;
        if(j < count) {
          if(j >= a_count) {
            local_keys[i] = keys_src.data[bb + j];
          } else {
            local_keys[i] = keys_src.data[a + j];
          }
        }
      }
    }

    workgroupBarrier();
  }

  fn reg_to_shared_strided(tid: u32) {
    for(var i = 0u; i < 15u; i = i + 1u) {
      shared_[128u * i + tid] = local_keys[i];
    }
    workgroupBarrier();
  }

  fn load_two_streams_shared(a_begin: u32, a_count: u32, b_begin: u32, b_count: u32, tid: u32) {
    // Load into register then make an unconditional strided store into memory.
    load_two_streams_reg(a_begin, a_count, b_begin, b_count, tid);
    reg_to_shared_strided(tid);
  }

   fn to_local(range: vec4<i32>) -> vec4<i32> {
    return vec4<i32>(
      0, 
      range.y-range.x, 
      range.y-range.x, 
      (range.y-range.x) + (range.w-range.z)
    );
  }


  fn merge_path_2(a_keys: i32, a_count: i32, b_keys: i32, b_count: i32, diag: i32) -> i32 {
    var begin = max(0, diag - b_count);
    var end   = min(diag, a_count);

    loop {
      if (begin >= end) {
        break;
      }
      
      let mid = u32(begin + end) / 2u;
      let a_key = shared_[u32(a_keys) + mid];
      let b_key = shared_[u32(b_keys) + u32(diag) - 1u - mid];

      if (!comp(b_key.x, a_key.x)) {
        begin = i32(mid + 1u);
      } else {
        end = i32(mid);
      }
    }

    return begin;
  }

  fn merge_path(range: vec4<i32>, diag: i32) -> i32 {
    return merge_path_2(
      range.x, 
      range.y - range.x, 
      range.z, 
      range.w - range.z,
      diag 
    );
  }
  
  fn segmented_merge_path(range: vec4<i32>, active_: vec2<i32>, diag: i32) -> vec3<i32> {
    // Consider a rectangle defined by range.
    // Now consider a sub-rectangle at the top-right corner defined by
    // active. We want to run the merge path only within this corner part.

    // If the cross-diagonal does not intersect our corner, return immediately.
    if (range.x + diag <= active_.x)  {
      return vec3<i32>(diag, active_.x, active_.y);
    }

    if (range.x + diag >= active_.y) {
      return vec3<i32>(range.y - range.x, active_.x, active_.y);
    } 

    // Call merge_path on the corner domain.
    var cactive = active_;
    cactive.x = max(cactive.x, range.x);
    cactive.y = min(cactive.y, range.w);

    let active_range = vec4<i32>(cactive.x, range.y, range.z, cactive.y);
    let active_offset = cactive.x - range.x;
    let p = merge_path(active_range, diag - active_offset);
    return vec3<i32>(p + active_offset, cactive.x, cactive.y);
  }

  fn partition_(range: vec4<i32>, mp0: i32, diag: i32) -> vec4<i32> {
    return vec4<i32>(range.x + mp0, range.y, range.z + diag - mp0, range.w);
  }

  fn segmented_serial_merge(range: vec4<i32>, active_: vec2<i32>) {
    var crange = range;
    crange.w = min(active_.y, crange.w);

    var a_key = shared_[crange.x];
    var b_key = shared_[crange.z];

    for(var i = 0u; i < 15u; i = i + 1u) {
      var p: bool;
      if (crange.x >= crange.y) {
        p = false;
      } else if (crange.z >= crange.w || crange.x < active_.x) {
        p = true;
      } else {
        p = !comp(b_key.x, a_key.x);
      }

      var index: u32 = u32(crange.x);
      if(!p) { 
        index = u32(crange.z); 
      }
      let c_key = shared_[index + 1u];
     
      if (p) {
        local_keys[i] = a_key;
        a_key = c_key;
        crange.x = i32(index + 1u);
      } else {
        local_keys[i] = b_key;
        b_key = c_key;
        crange.z = i32(index + 1u);
      }
    }
  }

  fn reg_to_shared_thread(tid: u32) {
    for(var i = 0u; i < 15u; i = i + 1u) {
      shared_[15u*tid+i] = local_keys[i];
    }
  }

  fn shared_to_mem(tid: u32, count: u32, first: u32) {
    if (count <= 128u * 15u) {
      for(var i = 0u; i < 15u; i = i + 1u) {
        let j = 128u * i + tid;
        keys_dst.data[first + j] = shared_[j];
      }
    } else {
      for(var i = 0u; i < 15u; i = i + 1u) {
        let j = 128u * i + tid;
        if (j < count) {
          keys_dst.data[first + j] = shared_[j];
        }
      }
    }
  }

  @compute @workgroup_size(128, 1, 1)
  fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
    @builtin(global_invocation_id) global_id: vec3<u32>,
  ) {
    let pass_ = (pass_counter.data / params.num_partition_ctas) - 1u;
    let nv = 15u * 128u;    
    let cta = workgroup_id.x;
    let tid = local_id.x;

    var range = merge_list.data[cta];

    let tile = range.w;
    let first = nv * u32(tile);
    let count2 = min(i32(nv), i32(params.count - first));
    range.w = range.z + (count2 - (range.y - range.x));

    let compressed_range = i32(compressed_ranges.data[u32(tile)]);
    var active_ = vec2<i32>(
      0x0000ffff & compressed_range,
      compressed_range >> 16u
    );
    
    load_two_streams_shared(
      u32(range.x), 
      u32(range.y - range.x), 
      u32(range.z), 
      u32(range.w - range.z), 
      tid
    );

    // Run a merge path search to find the starting point for each thread
    // to merge. If the entire warp fits into the already-sorted segments,
    // we can skip sorting it and leave its keys in shared memory.
    let list_parity = 1u & (u32(tile) >> pass_);
    if (list_parity != 0u) {
      active_ = vec2<i32>(0, active_.x);
    } else {
      active_ = vec2<i32>(active_.y, i32(nv));
    } 

    let warp_size = 32u;
    let warp_offset = 15u * (~(warp_size - 1u) & tid);
    var sort_warp: bool;
    if (list_parity != 0u)  {
      sort_warp = i32(warp_offset) < active_.y;
    } else {
      sort_warp = i32(warp_offset + 15u * warp_size) >= active_.x;
    }  
    
    for(var i = 0u; i < 15u; i = i + 1u) { local_keys[i] = vec2<u32>(0u); };
    let local_range = to_local(range);
    var mp = 0;
    var diag = 0u;
    var partitioned: vec4<i32>;

    workgroupBarrier();

    if (sort_warp) {
      diag = 15u * tid;
      let ret = segmented_merge_path(local_range, active_, i32(diag));
      mp = ret.x;
      //active_ = vec2<i32>(ret.y, ret.z);
      partitioned = partition_(local_range, i32(mp), i32(diag));
      segmented_serial_merge(partitioned, active_);
    }

    workgroupBarrier();
    
    if (sort_warp) {
      reg_to_shared_thread(tid);
    }

    workgroupBarrier();
    shared_to_mem(tid, u32(count2), first);
  }
)"