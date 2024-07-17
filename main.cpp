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
    const int iterations = 20;
  
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

  const uint32_t numQueries = 8;
  wgpu::Buffer querySetBuffer = utils::CreateBuffer(
    device, 
    numQueries * sizeof(uint64_t), 
    wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::QueryResolve,
    "QuerySet"
  );

  wgpu::QuerySetDescriptor querySetDescriptor;
  querySetDescriptor.count = numQueries;
  querySetDescriptor.label = "ComputeTime";
  querySetDescriptor.type = wgpu::QueryType::Timestamp;

  wgpu::QuerySet querySet = device.CreateQuerySet(&querySetDescriptor);

  sorter.Init(device, inputBuffer, maxCount, segmentsBuffer, maxNumSegments);
  for (uint32_t count = 2000000; count <= 2000000;  count += count / 10) {
    uint64_t cpu_time = 0;
    std::array<uint64_t, numQueries / 2> gpu_times;
    for(int i = 0; i < gpu_times.size(); i++) {
        gpu_times[i] = 0u;
    }

    int numSegments = ComputeUtil::div_up(count, 100);

    for (uint32_t it = 0; it < iterations; it++)  {
      std::vector<uint2> vec = ComputeUtil::fill_random_pairs(0, UINT32_MAX, count);
      std::vector<int> segments = ComputeUtil::fill_random_cpu(0, count - 1, numSegments, true);   

      device.GetQueue().WriteBuffer(inputBuffer, 0, vec.data(), vec.size() * sizeof(uint2));
      device.GetQueue().WriteBuffer(segmentsBuffer, 0, segments.data(), segments.size() * sizeof(int));
      sorter.Upload(device, count, numSegments);

      ComputeUtil::BusyWaitDevice(device);

      wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
      sorter.Sort(encoder, querySet, count, numSegments);

      encoder.ResolveQuerySet(querySet, 0, numQueries, querySetBuffer, 0);
      auto commandBuffer = encoder.Finish();

      auto t0 = high_resolution_clock::now();
      device.GetQueue().Submit(1, &commandBuffer);
      ComputeUtil::BusyWaitDevice(device);

      std::vector<uint64_t> queryData = ComputeUtil::CopyReadBackBuffer<uint64_t>(device, querySetBuffer, numQueries * sizeof(uint64_t));
      auto t1 = high_resolution_clock::now();

        for(int i = 0; i < numQueries / 2; i++) {
            gpu_times[i] += queryData[i * 2 + 1] - queryData[i * 2]; 
        }
    
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

    std::cout << count << " " << (cpu_time / iterations) / 1000 << std::endl;
    uint64_t tot_gpu = 0u;
    for (int i = 0; i < numQueries / 2; i++) {
        uint64_t v = ((gpu_times[i] / iterations) / 1000);
        std::cout << " +    " << v << std::endl;
        tot_gpu += v;
    }
    std::cout << " =    " << tot_gpu << std::endl;
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