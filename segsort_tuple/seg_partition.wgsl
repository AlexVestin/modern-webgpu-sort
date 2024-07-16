R"(
  struct Parameters {
    count: u32,
    nt: u32,
    vt: u32,
    nt2: u32,
    num_partitions: u32,
    num_segments: u32,
    num_ranges: u32,
    num_partition_ctas: u32,
    max_num_passes: u32,
  };

  const COPY_STATUS_OFFSET = 8192u;

  struct Data2 { data: array<vec2<u32>> };
  struct Data { data: array<u32> };
  struct AtomicData { data: array<atomic<i32>> };
  struct AtomicCounter { data: atomic<u32> };
  struct Ranges { data: array<vec2<i32>> };
  struct MergeRanges { data: array<vec4<i32>> };

  @binding(0) @group(0) var<storage, read> keys: Data2;
  @binding(1) @group(0) var<uniform> params: Parameters;
  @binding(2) @group(0) var<storage, read_write> source_ranges: Ranges;
  @binding(3) @group(0) var<storage, read> compressed_ranges: Data;
  @binding(4) @group(0) var<storage, read_write> pass_counter: AtomicCounter;
  @binding(5) @group(0) var<storage, read_write> op_counters: AtomicData;
  @binding(6) @group(0) var<storage, read_write> merge_list_data: MergeRanges;
  @binding(7) @group(0) var<storage, read_write> copy_list_data: Data;
  
  // 2*nt needed by scan
  var<workgroup> shared_: array<i32, 128>;

  fn comp(a_key: u32, b_key: u32) -> bool {
    return a_key < b_key;
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

  fn compute_mergesort_range_2(count: i32, partition_: i32, coop: i32, spacing: i32, mp0: i32, mp1: i32) -> vec4<i32> {
    var range = compute_mergesort_range(count, partition_, coop, spacing);
    let diag = spacing * partition_ - range.x;

    if(coop - 1 != ((coop - 1) & partition_)) {
      range.y = range.x + mp1;
      range.w = min(count, range.z + diag + spacing - mp1);
    }

    range.x = range.x + mp0;
    range.z = min(count, range.z + diag - mp0);
    return range;
  }

  fn merge_path_2(a_keys: i32, a_count: i32, b_keys: i32, b_count: i32, diag: i32) -> i32 {
    var begin = max(0, diag - b_count);
    var end   = min(diag, a_count);

    loop {
      if (begin >= end) {
        break;
      }
      
      let mid = u32(begin + end) / 2u;
      let a_key = keys.data[u32(a_keys) + mid];
      let b_key = keys.data[u32(b_keys) + u32(diag) - 1u - mid];

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
    return p + active_offset;
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

  fn scan(tid: u32, x: u32) -> vec2<i32> {
    var cx = i32(x);
    var first = 0u;

    shared_[first + tid] = cx;
    workgroupBarrier();
    // s_log2(64u)
    for(var i = 0u; i < 6u; i = i + 1u) {
      let offset = 1u << i;
      if (tid >= offset) {
        cx = cx + shared_[first + tid - offset];
      }
        
      first = 64u - first;
      shared_[first + tid] = cx;
      workgroupBarrier();
    }
      

    let count = 64u;
    var result: vec2<i32>;
    result.x = shared_[first + count - 1u];
    if (tid < count) {
      if (tid != 0u) {
        result.y = shared_[first + tid - 1u];
      } else {
        result.y = 0;
      }
    } else {
      result.y = result.x;
    }
    workgroupBarrier();

    return result;
  }

  fn get_range(pass_: u32) -> vec3<u32> {
    var range = params.num_ranges;
    var offset = 0u;
    var previous_offset = 0u;
    for (var i = 0u; i < pass_; i = i + 1u) {
      previous_offset = offset;
      range = (range + 1u) / 2u;
      offset = offset + range;
    }

    return vec3<u32>(range, offset, previous_offset);
  }


  @compute @workgroup_size(64, 1, 1)
  fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
  ) {

    let pass_ = atomicLoad(&pass_counter.data) / params.num_partition_ctas;
    let coop = 2 << pass_;

    if (local_id.x == 0u) {
      atomicAdd(&pass_counter.data, 1u);
    }

    let nv = 128u * 15u;    
    let spacing = nv;
    let cta = workgroup_id.x;
    let tid = local_id.x;

    let partition_ = (64u - 1u) * cta + tid;
    let first = nv * partition_;
    let count2 = min(nv, params.count - first);

    var mp0 = 0;
    let active_ = (tid < 64u - 1u) && (partition_ < params.num_partitions - 1u);
    let range_index = partition_ >> pass_;

    if (partition_ < params.num_partitions) {
      let range = compute_mergesort_range(i32(params.count), i32(partition_), i32(coop), i32(nv));
      let diag = min(i32(nv * partition_) - range.x, (range.y-range.x)+(range.w-range.z));

      let r = get_range(pass_);
      let indices = vec2<u32>( 
        min(r.x - 1u, ~1u & range_index), 
        min(r.x - 1u, 1u | range_index) 
      );

      var ranges: array<vec2<i32>, 2>;

      if(pass_ > 0u) {
        ranges[0] = source_ranges.data[r.z + indices.x];
        ranges[1] = source_ranges.data[r.z + indices.y];
      } else {
        for (var i = 0u; i < 2u; i = i + 1u) {
          let compressed = i32(compressed_ranges.data[indices[i]]);
          let first_r = i32(nv * indices[i]);

          ranges[i] = vec2<i32>(0x0000ffff & compressed, compressed >> 16u);
          if(i32(nv) != ranges[i].x) {
            ranges[i].x = ranges[i].x + first_r;
          } else{
            ranges[i].x = i32(params.count);
          } 
          if (-1 != ranges[i].y) {
            ranges[i].y = ranges[i].y + first_r;
          }
        }
      }

      let inner = vec2<i32>( 
        ranges[0].y, 
        max(range.z, ranges[1].x) 
      );
      let outer = vec2<i32> (
        min(ranges[0].x, ranges[1].x),
        max(ranges[0].y, ranges[1].y)
      );

      // Segmented merge path on inner.
      mp0 = segmented_merge_path(range, inner, diag);

      // Store outer merge range.
      if (active_ && 0 == diag) {
        source_ranges.data[r.y + range_index / 2u] = outer;
      }
    }

    shared_[tid] = mp0;
    workgroupBarrier();

    let mp1 = shared_[tid + 1u];
    workgroupBarrier();

    // Update the merge range to include partitioning.
    var range = compute_mergesort_range_2(i32(params.count), i32(partition_), i32(coop), i32(nv), i32(mp0), i32(mp1));

    // Merge if the source interval does not exactly cover the destination
    // interval. Otherwise copy or skip.
    var interval: vec2<i32>;
    if((1u & range_index) != 0u) {
      interval = vec2<i32>(range.z, range.w);
    }else {
      interval = vec2<i32>(range.x, range.y);
    } 

    var merge_op = false;
    var copy_op = false;

    // Create a segsort job.
    if (active_) {
      let interval_count = u32(interval.y-interval.x);
      merge_op = (first != u32(interval.x)) || (interval_count != count2);
      copy_op = !merge_op && (pass_ == 0u || copy_list_data.data[COPY_STATUS_OFFSET + partition_] == 0u);

      // Use the b_end component to store the index of the destination tile.
      // The actual b_end can be inferred from a_count and the length of 
      // the input array.
      range.w = i32(partition_);
    }

    let merge_scan = scan(tid, u32(merge_op));
    let copy_scan = scan(tid, u32(copy_op));

    if (tid == 0u) {
      shared_[0] = atomicAdd(&op_counters.data[pass_*6u], i32(merge_scan.x));
      shared_[1] = atomicAdd(&op_counters.data[(pass_*2u+1u) * 3u], i32(copy_scan.x));
    }
    workgroupBarrier();

    if (active_) {
      copy_list_data.data[COPY_STATUS_OFFSET + partition_] = u32(!merge_op);
      if (merge_op) {
        merge_list_data.data[u32(shared_[0]) + u32(merge_scan.y)] = range;
      }
        
      if (copy_op) {
        copy_list_data.data[u32(shared_[1]) + u32(copy_scan.y)] = partition_;
      }
    }
  }
)"