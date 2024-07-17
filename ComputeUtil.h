#pragma once

#include "wgpu/WGPUHelpers.h"
// #include "src/util/ImageUtil.h"
#include <random>
#include <iostream>
#include <vector>
#include <algorithm>
#include <string>

#include <thread>

struct int2 {
  int32_t x;
  int32_t y;
};

struct uint2 {
  uint32_t x;
  uint32_t y;
};

namespace ComputeUtil  {


template <typename T>
std::vector<T> ReadBackBuffer(const wgpu::Device& device, const wgpu::Buffer& fromBuffer, uint32_t byteSize) {
    WGPUBufferMapAsyncStatus readStatus = WGPUBufferMapAsyncStatus_Unknown;
    fromBuffer.MapAsync(
        wgpu::MapMode::Read, 0, byteSize,
        [](WGPUBufferMapAsyncStatus status, void* userdata) { *static_cast<bool*>(userdata) = status; }, &readStatus);

    uint32_t iterations = 0;
    while (readStatus == WGPUBufferMapAsyncStatus_Unknown) {
#ifndef __EMSCRIPTEN__
        device.Tick();
        std::this_thread::sleep_for(std::chrono::microseconds{50});
#endif
        if (iterations++ > 100000) {
            std::cout << " ------ Failed to retrieve buffer -------- " << std::endl;
            break;
        }
    }

    if (readStatus == WGPUBufferMapAsyncStatus_Success) {
        const T* data = static_cast<const T*>(fromBuffer.GetConstMappedRange());
        fromBuffer.Unmap();
        return {&data[0], &data[byteSize / sizeof(T)]};
    }

    std::cerr << "Failed to read back buffer, with status:" << static_cast<int>(readStatus) << std::endl;
    return {T()};
}

template <typename T>
  std::vector<T> CopyReadBackBuffer(const wgpu::Device& device, const wgpu::Buffer& fromBuffer, uint32_t byteSize) {
    wgpu::BufferDescriptor desc;
    desc.size = byteSize;
    desc.usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead;
    desc.mappedAtCreation = false;
    desc.label = "ReadbackBuffer";

    wgpu::Buffer copyBuffer = device.CreateBuffer(&desc);

    wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
    encoder.CopyBufferToBuffer(fromBuffer, 0, copyBuffer, 0, byteSize);
    wgpu::CommandBuffer commandBuffer = encoder.Finish();

    commandBuffer.SetLabel("ReadBackCommandBuffer");
    auto queue = device.GetQueue();
    queue.Submit(1, &commandBuffer);

    utils::BusyWaitDevice(device);

    std::vector<T> vv = std::move(ReadBackBuffer<T>(device, copyBuffer, byteSize));
    copyBuffer.Destroy();

    return vv;
  }

  wgpu::ComputePassEncoder CreateTimestampedComputePass(const wgpu::CommandEncoder& encoder, const wgpu::QuerySet& querySet, uint32_t index);

  wgpu::ComputePipeline CreatePipeline(
    const wgpu::Device& device, 
    const wgpu::BindGroupLayout& bgl, 
    const std::string& shader,
    const char* label
  ); 


  inline std::mt19937& get_mt19937();

  template <typename T>
  void PrintRange(
    const std::vector<T>& data, 
    int start, 
    int end, 
    uint32_t divisor = 16
  );

  std::vector<uint2> fill_random_pairs(
    int a, 
    int b, 
    size_t count
  );

  std::vector<int> fill_random_cpu(
    int a, 
    int b, 
    size_t count, 
    bool sorted = false
  );

  // Count leading zeros
  int clz(int x);
  constexpr bool is_pow2(int x);

  int32_t div_up(int32_t x, int32_t y);

  int find_log2(int x, bool round_up = false);

  void PrintGPUBuffer(
    const wgpu::Device& device, 
    const wgpu::Buffer& buffer, 
    uint32_t byteSize, 
    uint32_t newLineCount = 16
  );

  void BusyWaitDevice(
    const wgpu::Device& device
  );
} // namespace ComputeUtil

