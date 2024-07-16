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
  struct Data2 { data: array<vec2<u32>> };

  @binding(0) @group(0) var<storage, read> keys_src: Data2;
  @binding(1) @group(0) var<storage, read_write> keys_dst: Data2;
  @binding(2) @group(0) var<storage, read> copy_list: Data;
  @binding(3) @group(0) var<uniform> params: Parameters;

  var<private> local_keys: array<vec2<u32>, 15>;

  fn load_to_reg(tid: u32, count: u32, first: u32) {
    if(count >= 128u * 15u) {
      for(var i = 0u; i < 15u; i = i + 1u) {
        local_keys[i] = keys_src.data[first + 128u * i + tid];
        // keys_dst.data[first + 128u * i + tid] = keys_src.data[first + 128u * i + tid];
      }
    } else {
      for(var i = 0u; i < 15u; i = i +1u) {
        let j = 128u * i + tid;
        if(j < count){
          local_keys[i] = keys_src.data[first + j];
          //keys_dst.data[first + j] = keys_src.data[first + j];
        }
      }
    }
  } 

  fn store_to_mem(tid: u32, count: u32, first: u32) {
    if(count >= 128u * 15u) {
      for(var i = 0u; i < 15u; i = i + 1u) {
        keys_dst.data[first +  128u * i + tid] = local_keys[i];
      }
    } else {
      for(var i = 0u; i < 15u; i = i +1u) {
        let j = 128u * i + tid;
        if (j < count){
          keys_dst.data[first + j] = local_keys[i];
        }
      }
    }
  } 
 
  @compute @workgroup_size(128, 1, 1)
  fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
  ) {

    let nv = 128u * 15u;    
    let tile = copy_list.data[workgroup_id.x];
    let first = nv * tile;
    let count2 = min(i32(nv), i32(params.count - first));
    load_to_reg(local_id.x, u32(count2), first);
    store_to_mem(local_id.x, u32(count2), first);
  }
)"