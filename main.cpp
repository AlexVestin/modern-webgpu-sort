#include "SegSort.h"

#include <iostream>
#include "wgpu/NativeUtils.h"
#include "ComputeUtil.h"

#include <sstream>

static std::unique_ptr<wgpu::Instance> instance;

static const wgpu::BufferUsage storageUsage = wgpu::BufferUsage::Storage;
static const wgpu::BufferUsage copyDstUsage = storageUsage | wgpu::BufferUsage::CopyDst;
static const wgpu::BufferUsage copySrcUsage = storageUsage | wgpu::BufferUsage::CopySrc;
static const wgpu::BufferUsage copyAllUsage = copySrcUsage | copyDstUsage;

uint32_t unpack_x(const uint2& p) {
  return p.x & 0xffffu; 
}

uint32_t unpack_y(const uint2& p) {
  return p.x >> 16u; 
}

std::string PrintPos(const uint2& p) {
  std::stringstream ss;
  ss << "{ x: " << unpack_x(p) << " y: " << unpack_y(p) << " }";
  return ss.str(); 
}

void Test(const wgpu::Device& device) {
    const int iterations = 1;
  
  SegmentedSort sorter;
  const uint32_t maxCount = 12000000;
  int maxNumSegments = ComputeUtil::div_up(maxCount, 100);

  wgpu::Buffer inputBuffer = utils::CreateBuffer(
    device, 
    maxCount * sizeof(int2), 
    wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::CopySrc,
    "InputBuffer"
  );

  wgpu::Buffer segmentsBuffer = utils::CreateBuffer(
    device, 
    maxNumSegments * sizeof(int), 
    wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
    "SegmentsBuffer"
  );

  wgpu::Buffer querySetBuffer = utils::CreateBuffer(
    device, 
    2 * sizeof(uint64_t), 
    wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::QueryResolve,
    "QuerySet"
  );

  wgpu::QuerySetDescriptor querySetDescriptor;
  querySetDescriptor.count = 2;
  querySetDescriptor.label = "ComputeTime";
  querySetDescriptor.type = wgpu::QueryType::Timestamp;

  wgpu::QuerySet querySet = device.CreateQuerySet(&querySetDescriptor);

  sorter.Init(device, inputBuffer, maxCount, segmentsBuffer, maxNumSegments);
  for (uint32_t count = 1920; count < 8000000;  count += count / 10) {
    uint64_t cpu_time = 0;
    uint64_t gpu_time = 0;

    int numSegments = ComputeUtil::div_up(count, 100);

    for (uint32_t it = 0; it < iterations; it++)  {
      std::vector<uint2> vec = ComputeUtil::fill_random_pairs(0, UINT32_MAX, count);
      std::vector<int> segments = ComputeUtil::fill_random_cpu(0, count - 1, numSegments, true);   

      device.GetQueue().WriteBuffer(inputBuffer, 0, vec.data(), vec.size() * sizeof(uint2));
      device.GetQueue().WriteBuffer(segmentsBuffer, 0, segments.data(), segments.size() * sizeof(int));
      sorter.Upload(device, count, numSegments);

      ComputeUtil::BusyWaitDevice(device);

      wgpu::CommandEncoder encoder = device.CreateCommandEncoder();

      wgpu::ComputePassTimestampWrites writes;
      writes.beginningOfPassWriteIndex = 0u;
      writes.endOfPassWriteIndex = 1u;
      writes.querySet = querySet;

      wgpu::ComputePassDescriptor descriptor;
      descriptor.timestampWrites = &writes;
      wgpu::ComputePassEncoder computePass = encoder.BeginComputePass(&descriptor);

      sorter.Sort(device, computePass, count, numSegments);

      computePass.End();
      encoder.ResolveQuerySet(querySet, 0, 2, querySetBuffer, 0);
      auto commandBuffer = encoder.Finish();

      auto t0 = high_resolution_clock::now();
      device.GetQueue().Submit(1, &commandBuffer);
      ComputeUtil::BusyWaitDevice(device);

      std::vector<uint64_t> queryData = ComputeUtil::CopyReadBackBuffer<uint64_t>(device, querySetBuffer, 2 * sizeof(uint64_t));
      auto t1 = high_resolution_clock::now();


      gpu_time += queryData[1] - queryData[0];
      cpu_time += duration_cast<nanoseconds>(t1-t0).count();

      auto cmp = [](const uint2& a, const uint2& b) -> bool {
        return  a.x < b.x;
      };
      
      std::vector<uint2> output = ComputeUtil::CopyReadBackBuffer<uint2>(device, inputBuffer, count * sizeof(int2));
    

      std::vector<uint2> copy = vec;
      int cur = 0;
      for(int seg = 0; seg < segments.size(); seg++) {
        int next = segments[seg];
        std::sort(copy.data() + cur, copy.data() + next, cmp);
        cur = next;
      }
      std::sort(copy.data() + cur, copy.data() + vec.size(), cmp);

      for (int i = 0; i < output.size(); i++) {
        if (copy[i].x != output[i].x) {
          std::cerr << "Faulty at count " << count << " - i:" << i << ": " << PrintPos(output[i]) << " expected: " << PrintPos(copy[i]) << std::endl;
          exit(1);
        } 
      }
    }
    std::cout << count << " " << (cpu_time / iterations) / 1000 << " " << ((gpu_time / iterations) / 1000) << std::endl;
  }

  sorter.Dispose();
  inputBuffer.Destroy();
  segmentsBuffer.Destroy();
}

int main() {
    dawnProcSetProcs(&dawn::native::GetProcs());

    std::vector<const char*> enableToggleNames = { "allow_unsafe_apis" };
    std::vector<const char*> disabledToggleNames = { };

    wgpu::DawnTogglesDescriptor toggles;
    toggles.enabledToggles = enableToggleNames.data();
    toggles.enabledToggleCount = enableToggleNames.size();
    toggles.disabledToggles = disabledToggleNames.data();
    toggles.disabledToggleCount = disabledToggleNames.size();

    wgpu::InstanceDescriptor instanceDescriptor{};
    instanceDescriptor.nextInChain = &toggles;
    instanceDescriptor.features.timedWaitAnyEnable = true;
    instance = std::make_unique<wgpu::Instance>(wgpu::CreateInstance(&instanceDescriptor));

    if (instance == nullptr) { 
        std::cerr << "Failed to create instance" << std::endl;
        exit(1);
    }

    wgpu::Adapter adapter = NativeUtils::SetupAdapter(instance);
    wgpu::Device device = NativeUtils::SetupDevice(instance, adapter);

    Test(device);
    device.Destroy();
}