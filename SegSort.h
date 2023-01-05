#pragma once

#include <utility>
#include <chrono>
using namespace std::chrono;

#include "src/wgpu/webgpu.h"
#include "src/wgpu/WGPUHelpers.h"
#include "src/util/ImageUtil.h"


const int COPY_STATUS_OFFSET = 8192;

struct Param {
  uint32_t count;
  uint32_t nt;
  uint32_t vt;
  uint32_t nt2;
  uint32_t num_partitions;
  uint32_t num_segments;
  uint32_t num_ranges;
  uint32_t num_partition_ctas;
  uint32_t max_num_passes;
};

class SegmentedSort {
public:
    void Dispose();
    void CompileShaders(const wgpu::Device& device);

    void Init(
      const wgpu::Device& device,
      const wgpu::Buffer& inputBuffer, 
      uint32_t maxInputSize, 
      const wgpu::Buffer& segmentBuffer, 
      uint32_t maxSegmentSize
    );

    void Clear(const wgpu::CommandEncoder& encoder);

    void Upload(
        const wgpu::Device& device, 
        uint32_t count, 
        uint32_t segmentCount);


    void Sort(
        const wgpu::Device& device, 
        const wgpu::ComputePassEncoder& computePass,
        uint32_t count, 
        uint32_t segmentCount
    );

private:
    const uint32_t nt = 128;
    const uint32_t nt2 = 64;
    const uint32_t vt = 15;
    const uint32_t nv = 1920;

    uint32_t maxCount;
    uint32_t maxNumPasses;
    uint32_t maxNumSegments;
    uint32_t maxNumCtas;
    uint32_t maxCapacity;
    uint32_t previousCount = 0;

    void InitBuffers(const wgpu::Device& device);
    void InitClear(const wgpu::Device& device);
    void InitPartition(
        const wgpu::Device& device, 
        const wgpu::Buffer& inputBuffer
    );
    void InitCopy(
        const wgpu::Device& device, 
        const wgpu::Buffer& inputBuffer
    );
    void InitBlock(
        const wgpu::Device& device, 
        const wgpu::Buffer& inputBuffer, 
        const wgpu::Buffer& segmentsBuffer
    );
    void InitBinarySearch(
        const wgpu::Device& device, 
        const wgpu::Buffer& segmentsBuffer
    );
    void InitMerge(
        const wgpu::Device& device, 
        const wgpu::Buffer& inputBuffer
    );
    
    wgpu::Buffer inputBufferCopy;
    wgpu::Buffer paramBuffer;
    wgpu::Buffer partitionBuffer;
    wgpu::Buffer passCountBuffer;
    wgpu::Buffer compressedRangesBuffer;
    wgpu::Buffer mergeRangesBuffer;
    wgpu::Buffer copyListBuffer;
    wgpu::Buffer opCounterBuffer;
    wgpu::Buffer mergeListBuffer;

    wgpu::ComputePipeline blockPipeline;
    wgpu::ComputePipeline partitionPipeline;
    wgpu::ComputePipeline mergePipeline;
    wgpu::ComputePipeline binarySearchPipeline;
    wgpu::ComputePipeline copyPipeline;
    wgpu::ComputePipeline clearPipeline;

    wgpu::BindGroup copyBindGroups[2];
    wgpu::BindGroup binarySearchBindGroup;
    wgpu::BindGroup blockBindGroups[2];
    wgpu::BindGroup partitionBindGroups[2];
    wgpu::BindGroup mergeBindGroups[2];
    wgpu::BindGroup clearBindGroup;

    Param params;
};