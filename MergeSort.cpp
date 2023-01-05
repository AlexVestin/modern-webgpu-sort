#include <utility>

#include <chrono>
using namespace std::chrono;

#include "src/wgpu/webgpu.h"
#include "src/wgpu/NativeUtils.h"
#include "src/wgpu/WGPUHelpers.h"
#include "src/util/ImageUtil.h"
#include "ComputeUtil.h"

struct Param {
  uint32_t count;
  uint32_t nt;
  uint32_t vt;
  uint32_t coop;
  uint32_t num_partitions;
} params;

wgpu::Buffer inputBuffers[2];
wgpu::Buffer paramBuffer;
wgpu::Buffer partitionBuffer;
wgpu::Buffer passCountBuffer;

wgpu::ComputePipeline blockPipeline;
wgpu::ComputePipeline partitionPipeline;
wgpu::ComputePipeline mergePipeline;

wgpu::BindGroup blockBindGroups[2];
wgpu::BindGroup partitionBindGroups[2];
wgpu::BindGroup mergeBindGroups[2];


std::pair<int, int> GetTile(int cta, int nv, int count) {
  return { nv * cta, std::min(count, nv * (cta + 1)) };
}

void InitPartition(const wgpu::Device& device) {
  auto bgl = utils::MakeBindGroupLayout(
    device, {
        { 0, wgpu::ShaderStage::Compute, wgpu::BufferBindingType::Storage }, 
        { 1, wgpu::ShaderStage::Compute, wgpu::BufferBindingType::Uniform }, 
        { 2, wgpu::ShaderStage::Compute, wgpu::BufferBindingType::Storage }, 
        { 3, wgpu::ShaderStage::Compute, wgpu::BufferBindingType::Storage }, 
  });

  partitionPipeline = ComputeUtil::CreatePipeline(device, bgl,
     #include "mergesort/partition.wgsl"
     , "Sort::partition"
  );

  partitionBindGroups[0] = utils::MakeBindGroup(
    device, bgl,
        {
          { 0, inputBuffers[0] },
          { 1, paramBuffer },
          { 2, partitionBuffer },
          { 3, passCountBuffer }
    });
  
  partitionBindGroups[1] = utils::MakeBindGroup(
    device, bgl,
        {
          { 0, inputBuffers[1] },
          { 1, paramBuffer },
          { 2, partitionBuffer },
          { 3, passCountBuffer }
    });
}

void InitBlock(const wgpu::Device& device) {
  auto bgl = utils::MakeBindGroupLayout(
    device, {
        { 0, wgpu::ShaderStage::Compute, wgpu::BufferBindingType::Storage },
        { 1, wgpu::ShaderStage::Compute, wgpu::BufferBindingType::Storage },
        { 2, wgpu::ShaderStage::Compute, wgpu::BufferBindingType::Uniform }, 
  });

  blockPipeline = ComputeUtil::CreatePipeline(device, bgl,
    #include "mergesort/block.wgsl"
    , "Sort::blockPipeline"
  );

  blockBindGroups[0] = utils::MakeBindGroup(
    device, bgl,
        {
          { 0, inputBuffers[0] },
          { 1, inputBuffers[0] },
          { 2, paramBuffer },
    });
  blockBindGroups[1] = utils::MakeBindGroup(
    device, bgl,
        {
          { 0, inputBuffers[0] },
          { 1, inputBuffers[1] },
          { 2, paramBuffer },
    });
} 

void InitMerge(const wgpu::Device& device) {
  auto bgl = utils::MakeBindGroupLayout(
    device, {
        { 0, wgpu::ShaderStage::Compute, wgpu::BufferBindingType::ReadOnlyStorage },
        { 1, wgpu::ShaderStage::Compute, wgpu::BufferBindingType::Storage },
        { 2, wgpu::ShaderStage::Compute, wgpu::BufferBindingType::Uniform }, 
        { 3, wgpu::ShaderStage::Compute, wgpu::BufferBindingType::Storage }, 
        { 4, wgpu::ShaderStage::Compute, wgpu::BufferBindingType::ReadOnlyStorage }, 
  });

  mergePipeline = ComputeUtil::CreatePipeline(device, bgl,
    #include "mergesort/merge.wgsl"
    , "Sort::mergePipeline"
  );
 
  mergeBindGroups[0] = utils::MakeBindGroup(
    device, bgl,
        {
          { 0, inputBuffers[0] },
          { 1, inputBuffers[1] },
          { 2, paramBuffer },
          { 3, partitionBuffer },
          { 4, passCountBuffer }
    });

  mergeBindGroups[1] = utils::MakeBindGroup(
    device, bgl,
        {
          { 0, inputBuffers[1] },
          { 1, inputBuffers[0] },
          { 2, paramBuffer },
          { 3, partitionBuffer },
          { 4, passCountBuffer }
    });
} 

void MergeSort(const wgpu::Device& device, int count) {
  int nv = 128 * 15;
  int num_ctas = ComputeUtil::div_up(count, nv);
  int num_passes = ComputeUtil::find_log2(num_ctas, true);

  uint8_t blockBindgroupIndex = (1 & num_passes) ? 1 : 0;
  uint32_t mergeBindgroupIndex = 0;

  int num_partitions = num_ctas + 1;
  uint32_t numPartitionDispatch = ComputeUtil::div_up(num_partitions, nv);
  params.nt = 128;
  params.vt = 15;
  params.count = count;
  params.num_partitions = num_partitions;
  params.coop = numPartitionDispatch;
  device.GetQueue().WriteBuffer(paramBuffer, 0, &params, sizeof(Param));
  auto encoder = device.CreateCommandEncoder();
  auto computePass = encoder.BeginComputePass();

  uint32_t numSortDispatch = num_ctas;
  computePass.SetPipeline(blockPipeline);
  computePass.SetBindGroup(0, blockBindGroups[blockBindgroupIndex]);
  computePass.Dispatch(numSortDispatch);

  if (1 & num_passes) {
    mergeBindgroupIndex++;
  }

  for (int pass = 0; pass < num_passes; pass++) {
    computePass.SetPipeline(partitionPipeline);
    computePass.SetBindGroup(0, partitionBindGroups[mergeBindgroupIndex % 2]); 
    computePass.Dispatch(numPartitionDispatch);

    computePass.SetPipeline(mergePipeline);
    computePass.SetBindGroup(0, mergeBindGroups[mergeBindgroupIndex % 2]); 
    computePass.Dispatch(numSortDispatch);

    mergeBindgroupIndex++;
  }

  computePass.EndPass();
  auto commandBuffer = encoder.Finish();
  device.GetQueue().Submit(1, &commandBuffer); 
}

int main() {
  wgpu::Device device = CreateCppDawnDevice(nullptr);

  uint32_t spacing = 128;
  for (uint32_t count = 2000; count < 90000000;  count += count/10) {
    uint64_t gpu_time = 0;
    for (uint32_t it = 0; it < 5; it++)  {
      auto vec = ComputeUtil::fill_random_cpu(0, 100000, count);
    
      inputBuffers[0] = utils::CreateBufferFromData(
        device, 
        vec.data(), 
        vec.size() * sizeof(int), 
        wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc 
      );

      inputBuffers[1] = utils::CreateBuffer(
        device, 
        vec.size() * sizeof(int), 
        wgpu::BufferUsage::Storage
      );

      paramBuffer = utils::CreateBufferFromData(
        device, 
        &params, 
        sizeof(Param), 
        wgpu::BufferUsage::Uniform
      );

      partitionBuffer = utils::CreateBuffer(
        device,
        sizeof(int) * (ComputeUtil::div_up(count, 128*15) + 1),
        wgpu::BufferUsage::Storage 
      );
       
      passCountBuffer = utils::CreateBuffer(
        device,
        sizeof(int),
        wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::CopySrc 
      );

      InitBlock(device);
      InitPartition(device);
      InitMerge(device);

      bool done = false;

      device.GetQueue().OnSubmittedWorkDone(0, [](WGPUQueueWorkDoneStatus status, void * userdata) -> void {
        *static_cast<bool*>(userdata) = true;
      }, &done);

      while (!done) {
        device.Tick();
      }

      auto t0 = high_resolution_clock::now();
      MergeSort(device, count);
      done = false;
      device.GetQueue().OnSubmittedWorkDone(0, [](WGPUQueueWorkDoneStatus status, void * userdata) -> void {
        *static_cast<bool*>(userdata) = true;
      }, &done);

      while (!done) {
        device.Tick();
      }

      auto t1 = high_resolution_clock::now();
      gpu_time += duration_cast<nanoseconds>(t1-t0).count();
      std::vector<int> output = CopyReadBackBuffer<int>(
        device, 
        inputBuffers[0], 
        vec.size() * sizeof(int)
      );

      bool bad = false;
      int i = 1;
      for (; i<  output.size(); i++) {
        if(output[i] < output[i-1]) {
          bad = true;
          break;
        }
      }
      
      blockBindGroups[0].Release();
      blockBindGroups[1].Release();
      partitionBindGroups[0].Release();
      partitionBindGroups[1].Release();
      mergeBindGroups[0].Release();
      mergeBindGroups[1].Release();

      blockPipeline.Release();
      partitionPipeline.Release();
      mergePipeline.Release();

      inputBuffers[0].Destroy();
      inputBuffers[1].Destroy();
      partitionBuffer.Destroy();
      passCountBuffer.Destroy();
      paramBuffer.Destroy();

      if (bad) {
        std::cerr << "Faulty at: " << count  << std::endl;
        return 1;
      }
    }
    
    std::cout << count << " " << (gpu_time / 5) / 1000 << std::endl;
  }

  return 0;
}
