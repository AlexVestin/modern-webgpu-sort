#include "Subgroups.h"

#include "ComputeUtil.h"

struct UniformData {
  uint32_t count;
  uint32_t step;
  uint32_t padding0;
  uint32_t padding1;
};

void SubgroupSort::Init(const wgpu::Device& device, const wgpu::Buffer& inputBuffer, uint32_t inputSize) {
    auto bgl = utils::MakeBindGroupLayout(
    device, "SubgroupsSort", {
        { 0, wgpu::ShaderStage::Compute, wgpu::BufferBindingType::Storage },
        { 1, wgpu::ShaderStage::Compute, wgpu::BufferBindingType::Uniform },
    });

  pipeline = ComputeUtil::CreatePipeline(device, bgl,
    #include "subgroups/sort.wgsl"
    , "Sort::Subgroups"
  );

  uniformBuffer = utils::CreateBuffer(device, sizeof(UniformData), wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst, "SubgroupUniforms");

  bindGroup = utils::MakeBindGroup(
    device, bgl,
        {
          { 0, inputBuffer },
          { 1, uniformBuffer }
    });
}

void SubgroupSort::Upload(const wgpu::Device& device, uint32_t count) {
    uint32_t numWgs = ComputeUtil::div_up(count, 32);
   
    UniformData data;
    data.count = count;

    if (numWgs >= 0xffffu) {
      uint32_t x = static_cast<uint32_t>(std::sqrt(static_cast<double>(numWgs)));
      data.step = x;
    } else {
      data.step = 1u;
    }

    device.GetQueue().WriteBuffer(uniformBuffer, 0, &data, sizeof(UniformData));
}


void SubgroupSort::Sort(const wgpu::CommandEncoder& encoder, const wgpu::QuerySet& querySet, uint32_t count) {
    auto sortPass = ComputeUtil::CreateTimestampedComputePass(encoder, querySet, 0);
    
    uint32_t numWgs = ComputeUtil::div_up(count, 32);

    sortPass.SetPipeline(pipeline);
    sortPass.SetBindGroup(0, bindGroup);

  if (numWgs >= 0xffffu) {
      uint32_t x = static_cast<uint32_t>(std::sqrt(static_cast<double>(numWgs)));
      uint32_t y = x + 1;
      sortPass.DispatchWorkgroups(x, y);
    } else {
      sortPass.DispatchWorkgroups(numWgs);
    }
    
    sortPass.End();
}

void SubgroupSort::Dispose() {

}