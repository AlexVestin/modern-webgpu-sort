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
    max_num_passes: u32,
  };

  struct Data { data: array<u32> };
  struct AtomicCounter { data: u32 };

  @binding(0) @group(0) var<uniform> params: Parameters;
  @binding(1) @group(0) var<storage, read_write> op_counters: Data;
  @binding(2) @group(0) var<storage, read_write> pass_counter: AtomicCounter;
 
  @compute @workgroup_size(128, 1, 1)
  fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(global_invocation_id) global_id: vec3<u32>
  ) {

    if (global_id.x == 0u) {
        pass_counter.data = 0u;
    }   

    let idx = workgroup_id.x * 128u + local_id.x;

    if (idx < params.max_num_passes * 24u) {
        // 0 if divisible by 3, 1 otherwise
        op_counters.data[idx] = ((idx % 3u) + 1u) / 2u;
    }
  }
)"