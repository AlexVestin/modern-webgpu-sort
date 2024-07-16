R"(
  struct Parameters {
    count: u32,
    nt: u32,
    vt: u32,
    num_wg: u32,
    num_partitions: u32
  };

  struct Data { data: array<u32> };
  struct Counter { data: atomic<u32> };

  @binding(0) @group(0) var<storage, read_write> keys: Data;
  @binding(1) @group(0) var<uniform> params: Parameters;
  @binding(2) @group(0) var<storage, read_write> partitions: Data;
  @binding(3) @group(0) var<storage, read_write> counter: Counter;

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

  fn merge_path_2(a_keys: i32, a_count: i32, b_keys: i32, b_count: i32, diag: i32) -> u32 {
    var begin = max(0, i32(diag - b_count));
    var end   = min(diag, a_count);

    loop {
      if (begin >= end) {
        break;
      }
      let mid = (begin + end) / 2;
      let a_key = keys.data[u32(a_keys + mid)];
      let b_key = keys.data[u32(b_keys + diag - 1 - mid)];

      if (a_key <= b_key) {
        begin = mid + 1;
      } else {
        end = mid;
      }
    }

    return u32(begin);
  }

  @compute @workgroup_size(128, 1, 1)
  fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
  ) {

    let nv = 128u * 15u;    

    let pass_ = atomicLoad(&counter.data) / params.num_wg;
    let coop = 2 << pass_;

    if (local_id.x == 0u) {
      atomicAdd(&counter.data, 1u);
    }

    let spacing = i32(nv);

    for (var i = 0u; i < 15u; i = i + 1u) {
      let index = workgroup_id.x * nv + i * 128u + local_id.x;

      let range = compute_mergesort_range(i32(params.count), i32(index), i32(coop), spacing);
      let diag  = min(spacing * i32(index), i32(params.count)) - range.x;
      let path = merge_path_2(
        range.x, 
        range.y - range.x, 
        range.z, 
        range.w - range.z, 
        diag
      );
      
      if (index < params.num_partitions) {
        partitions.data[index] = u32(path);
      }
    }
  }
)"