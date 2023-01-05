#pragma once

#include "src/wgpu/WGPUHelpers.h"
#include "src/util/ImageUtil.h"
#include <random>
#include <iostream>
#include <vector>
#include <algorithm>
#include <string>

struct Pair {
  int32_t x, y;
};

namespace ComputeUtil  {
  wgpu::ComputePipeline CreatePipeline(
    const wgpu::Device& device, 
    const wgpu::BindGroupLayout& bgl, 
    const std::string& shader,
    const char* label
  ); 

  void CreatePipelineFromFile(
    const wgpu::Device& device, 
    const wgpu::BindGroupLayout& bgl, 
    const std::string& filename,
    wgpu::ComputePipeline* pl
  );

  inline std::mt19937& get_mt19937();

  template <typename T>
  void PrintRange(
    const std::vector<T>& data, 
    int start, 
    int end, 
    uint32_t divisor = 16
  );

  std::vector<Pair> fill_random_pairs(
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

