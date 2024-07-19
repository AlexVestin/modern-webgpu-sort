

R"(
  enable subgroups;

  struct UniformData {
    count: u32,
    step: u32,
  }

  @binding(0) @group(0) var<storage, read_write> data: array<u32>;
  @binding(1) @group(0) var<uniform> uniforms: UniformData;
  
  fn getLaneMaskLt(idx: u32) -> u32 {
      return (1u << idx) - 1u;
  }

  const WG_SIZE = 32u;

  var<workgroup> shared_mem: array<u32, 32>;
 
  @compute @workgroup_size(WG_SIZE, 1, 1)
  fn main(
    @builtin(global_invocation_id) global_invocation_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(subgroup_size) sg_size : u32,
    @builtin(subgroup_invocation_id) sg_id : u32
  ) {
    let base = (wg_id.y * uniforms.step + wg_id.x) * WG_SIZE;
    if (base >= uniforms.count) {
       return;
    }

    var geMask = getLaneMaskLt(sg_id);
    let key = data[global_invocation_id.x];

    for (var bit = 0u; bit < 32u; bit++) {
        let currentBit = 1u << bit;
        let isBitNotSet = (key & currentBit) == 0u;
        let ballot = subgroupBallot2(isBitNotSet)[0u];
        
        if (isBitNotSet) {
            geMask &= ballot;
        } else {
            geMask |= ballot;
        }
    }

    let scatterTo = countOneBits(geMask);
    data[base + scatterTo] = key;    
  }
)"


