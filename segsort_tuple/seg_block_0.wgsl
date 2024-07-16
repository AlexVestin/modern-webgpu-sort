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

  const words_per_thread = 4u;

  struct Data2 { data: array<vec2<u32>> };
  struct Data { data: array<u32> };

  @binding(0) @group(0) var<storage, read_write> keys_src: Data2;
  @binding(1) @group(0) var<uniform> params: Parameters;
  @binding(2) @group(0) var<storage, read> segments: Data;
  @binding(3) @group(0) var<storage, read> partitions: Data;
  @binding(4) @group(0) var<storage, read_write> compressedRanges: Data;

  // nt * vt (128 * 15) + 1
  var<workgroup> shared_: array<vec2<u32>, 1921>;
  var<workgroup> ranges: array<i32, 128>;
  var<private> local_keys: array<vec2<u32>, 15>;

  fn s_log2(x: u32) -> u32 {
    if (x <= 1u) { return 0u; }

    var v = x;
    var c = 0u;
    loop { 
      v = v / 2u; 
      if (v == 0u) { break; } 
      c = c + 1u; 
    }
    return c;
  }
  
  fn comp(a_key: u32, b_key: u32) -> bool {
    return a_key < b_key;
  }

  fn mem_to_reg_strided(global_offset: u32, tid: u32, count: u32) {
    if (count >= 128u * 15u) {
      for (var i = 0u; i < 15u; i = i + 1u) {
        local_keys[i] = keys_src.data[global_offset + 128u*i+tid];
      }
    } else {
      for (var i = 0u; i < 15u; i = i + 1u) {
        let j = 128u * i + tid;
        if(j < count) {
          local_keys[i] = keys_src.data[global_offset + 128u*i+tid];
        }
      }   
    }
  }

  fn bfi(x: u32, y: u32, bit: u32, num_bits: u32) -> u32 {
    var result: u32;
    var num_bits_c = num_bits;
    if(bit + num_bits_c > 32u) {
      num_bits_c = 32u - bit;
    } 
    var mask = ((1u << num_bits_c) - 1u) << bit;
    result = y & ~mask;
    result = result | (mask & (x << bit));
    return result;
  }

  fn ffs(x: u32) -> u32 {
    for (var i = 0u; i < 32u; i = i + 1u) {
      if (((1u << i) & x) != 0u) { return i + 1u; } 
    }
      
    return 0u;
  }

  fn clz(x: u32) -> u32 {
    for(var i = 31u; i >= 0u; i = i - 1u) {
      if(((1u << i) & x) != 0u) { return 31u - i; }
    }
      
    return 32u;
  }

  fn reg_to_shared_strided(tid: u32) {
    for(var i = 0u; i < 15u; i = i + 1u) { shared_[128u * i + tid] = local_keys[i]; }
  }

  fn shared_to_reg_thread(tid: u32) {
    for (var i = 0u; i < 15u; i = i +1u) { local_keys[i] = shared_[15u * tid + i]; }
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
    if (count >= 128u * 15u) {
      for (var i = 0u; i < 15u; i = i + 1u) {
        keys_src.data[global_offset + 128u*i+tid] = local_keys[i];
      }
    } else {
      for (var i = 0u; i < 15u; i = i + 1u) {
        let j = 128u * i + tid;
        if (j < count) {
          keys_src.data[global_offset + j] = local_keys[i];
        }
      }   
    }
  }

  fn reg_to_mem_thread(global_offset: u32, tid: u32, count: u32) {
    reg_to_shared_thread(tid);
    shared_to_reg_strided(tid);
    reg_to_mem_strided(global_offset, tid, count);
  }

  // TODO: unsigned / signed?
  fn out_of_range_flags(first: u32, vt: u32, count: u32) -> u32 {
    let out_of_range = min(i32(vt), i32(first + vt - count));
    var head_flags = 0u;
    if (out_of_range > 0) {
      let mask = (1u << vt) - 1u;
      head_flags = mask & ( ~mask >> u32(out_of_range));
    }
    return head_flags;
  }

  fn swap(i: u32, j: u32) {
    let temp = local_keys[i];
    local_keys[i] = local_keys[j];  
    local_keys[j] = temp;
  }

  fn odd_even_sort(flags: u32) {
    for(var j = 0u; j < 15u; j = j + 1u) {
      for (var i = 1u & j; i < 15u - 1u; i = i + 2u) {
        if((0u == ((2u << i) & flags)) && comp(local_keys[i + 1u].x, local_keys[i].x)) {
          swap(i, i + 1u);
        }
      }
    }
  }

  fn get_tile(cta: u32, nv: u32, count: u32) -> vec2<u32> {
    return vec2<u32>(nv * cta, min(count, nv * (cta + 1u)));
  }

  fn prmt(a: u32, b: u32, index: u32) -> u32 {
    var result: u32 = 0u;

    for(var i = 0u; i < 4u; i = i + 1u) {
      let sel = 0xfu & (index >> (4u * i));
      var x = a;
      if ((7u & sel) > 3u) { 
        x = b; 
      };
      x = 0xffu & (x >> (8u * (3u & sel)));
      if ((8u & sel) != 0u) {
        if ((128u & x) != 0u) {
          x = 0xffu;
        } else {
          x = 0u;
        }
      } 
      result = result | (x << (8u * i));
    }
    return result;
  }

  fn load(partitions_global: vec2<u32>, nv: u32, tid: u32, cta: u32, count: u32) -> u32 {

    let mp0 = partitions_global[0];
    let mp1 = partitions_global[1];
    let gid = nv * cta;
    let gid_count = count - gid;

    // Set the head flags for out-of-range keys.
    var head_flags = out_of_range_flags(15u * tid, 15u, gid_count);

    if (mp1 > mp0) {
      // Clear the flag bytes, then loop through the indices and poke in
      // flag bytes.
      for (var i = 0u; i < words_per_thread; i = i + 1u) {
        shared_[128u * i + tid].x = 0u;
      }
      workgroupBarrier();

     
      // Workaround for no union 8 bit shared storage
      let mpl = (u32(mp1 - mp0) + 127u) / 128u;
      let base = u32(mp0) + mpl * tid;
      var prev_val: u32 = 100000000u;
      if (base > 0u && base < mp1) {
        prev_val = u32(segments.data[base - 1u] - gid);
      }
      var mp_count = 0u;
      var has_own = false;
      loop {
        let index = base + mp_count;
        if (index >= mp1) { 
          break; 
        }
        
        let cur = u32(segments.data[index] - gid);
        let same_slot = prev_val / 4u == cur / 4u; 
        if (same_slot && !has_own) {
          mp_count = mp_count + 1u;
          continue;
        } 

        if (!same_slot && mp_count >= mpl) {
          break;
        }
        
        let flag_index = cur / 4u;
        // TODO: validate parenthesis
        let val = 1u << ((cur % 4u) * 8u); 
        shared_[flag_index].x = shared_[flag_index].x | val;
        has_own = true;
        mp_count = mp_count + 1u;
        prev_val = cur;
      };

      workgroupBarrier();

      // Combine all the head flags for this thread.
      let first = 15u * tid;
      let offset = first / 4u;
      var prev = shared_[offset].x;
      let mask = 0x3210u + 0x1111u * (3u & first);

      for(var i = 0u; i < words_per_thread; i = i + 1u) {
        let next = shared_[offset + 1u + i].x;
        let x = prmt(prev, next, mask);
        prev = next;

        // Set the head flag bits.
        if ((0x00000001u & x) != 0u){ head_flags = head_flags | (1u << (4u * i + 0u)); }
        if ((0x00000100u & x) != 0u){ head_flags = head_flags | (1u << (4u * i + 1u)); }
        if ((0x00010000u & x) != 0u){ head_flags = head_flags | (1u << (4u * i + 2u)); }
        if ((0x01000000u & x) != 0u){ head_flags = head_flags | (1u << (4u * i + 3u)); }
      }

      head_flags = head_flags & ((1u << 15u) - 1u);

      workgroupBarrier();
    }

    return head_flags;
  }
  
  fn partition_(range: vec4<i32>, mp0: i32, diag: i32) -> vec4<i32> {
    return vec4<i32>(range.x + mp0, range.y, range.z + diag - mp0, range.w);
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

  fn compute_mergesort_range(count: i32, partition_: i32, coop: i32, spacing: i32) -> vec4<i32> {
    let frame = compute_mergesort_frame(partition_, coop, spacing);
    return vec4<i32>(
      frame.x,
      min(count, frame.y),
      min(count, frame.z),
      min(count, frame.w)
    );
  }

  fn merge_path_2(a_keys: i32, a_count: i32, b_keys: i32, b_count: i32, diag: i32) -> u32 {
    var begin = max(0, diag - b_count);
    var end   = min(diag, a_count);

    loop {
      if (begin >= end) {
        break;
      }
      let mid = u32(begin + end) / 2u;
      let a_key = shared_[u32(a_keys) + mid];
      let b_key = shared_[u32(b_keys + diag) - 1u - mid];

      if (!comp(b_key.x, a_key.x)) {
        begin = i32(mid + 1u);
      } else {
        end = i32(mid);
      }
    }

    return u32(begin);
  }

  fn merge_path(range: vec4<i32>, diag: i32) -> u32 {
    return merge_path_2(
      range.x, 
      range.y - range.x, 
      range.z, 
      range.w - range.z,
      diag 
    );
  }

  fn segmented_merge_path(range: vec4<i32>, active_: vec2<i32>, diag: i32) -> i32 {
    // Consider a rectangle defined by range.
    // Now consider a sub-rectangle at the top-right corner defined by
    // active. We want to run the merge path only within this corner part.

    // If the cross-diagonal does not intersect our corner, return immediately.
    if (range.x + diag <= active_.x)  {
      return diag;
    }

    if (range.x + diag >= active_.y) {
      return range.y - range.x;
    } 

    // Call merge_path on the corner domain.
    var cactive = active_;
    cactive.x = max(cactive.x, range.x);
    cactive.y = min(cactive.y, range.w);

    let active_range = vec4<i32>(cactive.x, range.y, range.z, cactive.y);

    let active_offset = cactive.x - range.x;
    let p = merge_path(active_range, diag - active_offset);

    return i32(p) + active_offset;
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
      if(!p) { index = u32(crange.z); }
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
  
    workgroupBarrier();
  }

  fn merge_pass(tid: u32, count: u32, pass_: u32, active_: vec2<i32>) -> vec2<i32> {
    var cactive = active_;

    let list = i32(tid >> pass_);
    // Fetch the active range for the list this thread's list is merging with.
    let sibling_range = ranges[1 ^ list];
    let sibling = vec2<i32>(i32(0x0000ffff & sibling_range), i32(sibling_range >> 16u));

    let list_parity = 1 & list;
    // This pass does a segmented merge on ranges list and 1 ^ list.
    // ~1 & list is the left list and 1 | list is the right list.
    // We find the inner segments for merging, then update the active
    // range to the outer segments for the next pass.
    var left: vec2<i32>;
    var right: vec2<i32>;
    if (list_parity != 0) {
      left = sibling;
      right = cactive;
    } else {
      left = cactive;
      right = sibling;
    }

    let inner = vec2<i32>(left.y, right.x);
    cactive.x = min(left.x, right.x);
    cactive.y = max(left.y, right.y);

    // Store the data from thread order into shared memory.
    reg_to_shared_thread(tid);
    let coop = 2 << pass_;
    let range = compute_mergesort_range(i32(count), i32(tid), i32(coop), i32(15u));
    let diag = 15u * tid - u32(range.x);
    let mp = segmented_merge_path(range, inner, i32(diag));

    // Run a segmented serial merge.
    let part = partition_(range, i32(mp), i32(diag));
    segmented_serial_merge(part, inner);
    
    // Pack and store the outer range to shared memory.
    ranges[list >> 1u] = i32(bfi(u32(cactive.y), u32(cactive.x), 16u, 16u));
    workgroupBarrier();

    return cactive;
  }

  fn block_sort(tid: u32, count: u32, head_flags: u32) -> vec2<i32> {
  
    // Sort the inputs within each thread.
    odd_even_sort(head_flags);

    // Record the first and last occurrences of head flags in this segment.
    var active_: vec2<i32>;
    if (head_flags != 0u) {
      active_.x = i32(15u * tid) - 1 + i32(ffs(head_flags));
      active_.y = i32(15u * tid) + 31 - i32(clz(head_flags));
    } else {
      active_.x = i32(15u * 128u);
      active_.y = -1;
    }

    ranges[tid] = i32( bfi(u32(active_.y), u32(active_.x), 16u, 16u) );
    workgroupBarrier();

    let num_passes = s_log2(128u);
    // Merge threads starting with a pair until all values are merged.
    for (var pass_ = 0u; pass_ < num_passes; pass_++) {
      active_ = merge_pass(tid, count, pass_, active_);
    }

    return active_;
  }

  @compute @workgroup_size(128, 1, 1)
  fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(global_invocation_id) global_id: vec3<u32>
  ) {
    let nv = 15u * 128u;
    let tile = get_tile(workgroup_id.x, nv, params.count);
    let tile_count = tile.y - tile.x;

    let p = vec2<u32>(
      partitions.data[workgroup_id.x], 
      partitions.data[workgroup_id.x + 1u]
    );
    let head_flags = load(p, nv, local_id.x, workgroup_id.x, params.count);
    mem_to_reg_thread(tile.x, local_id.x, tile_count);
    let active_ = block_sort(local_id.x, tile_count, head_flags);

    reg_to_mem_thread(tile.x, local_id.x, tile_count);

    // segmented partitioning kernels.
    if (local_id.x == 0u) {
     compressedRanges.data[workgroup_id.x] = bfi(u32(active_.y), u32(active_.x), 16u, 16u);
    }
  }
)"