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

  struct Data { data: array<u32> };

  @binding(0) @group(0) var<storage, read> segments: Data;
  @binding(1) @group(0) var<storage, read_write> partitions: Data;
  @binding(2) @group(0) var<uniform> params: Parameters;

  fn binary_search(count: u32, key: u32) -> u32 {
    var begin = 0u;
    var end = count;

    loop {
      if (begin >= end) {
        break;
      }

      let mid = (begin + end) / 2u;
      let key2 = segments.data[mid];
      if (key2 < key) {
        begin = mid + 1u;
      } else {
        end = mid;
      }
    };

    return begin;
  }

 
  @compute @workgroup_size(128, 1, 1)
  fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
  ) {

    let nv = 128u * 15u;    
    let spacing = nv;

    let base = workgroup_id.x * nv;
    for (var i = base + local_id.x; i < params.num_partitions; i = i + 128u) {
      let key = min(spacing * i, params.count);
      partitions.data[i] = binary_search(params.num_segments, key);
    }
  }
)"