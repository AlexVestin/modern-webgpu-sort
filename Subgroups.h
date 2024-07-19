#pragma once

#include <utility>
#include <chrono>
using namespace std::chrono;

#include <webgpu/webgpu_cpp.h>
#include "wgpu/WGPUHelpers.h"


class SubgroupSort {
public:
    void Dispose();
    void Init(
      const wgpu::Device& device,
      const wgpu::Buffer& inputBuffer, 
      uint32_t inputSize
    );

    void Upload(const wgpu::Device& device, uint32_t count);
    void Sort(const wgpu::CommandEncoder& encoder, const wgpu::QuerySet& querySet, uint32_t count);
private:
    wgpu::ComputePipeline pipeline;
    wgpu::BindGroup bindGroup;
    wgpu::Buffer uniformBuffer;
};