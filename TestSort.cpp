#include "SegSort.h"
#include "ComputeUtil.h"
#include "src/wgpu/NativeUtils.h"

#include <sstream>

int32_t unpack_x(const Pair& p) {
  return (p.x & 0xffff) - (0xffff + 1) * ((p.x & 0xffff) >> 15); 
}

std::string PrintPos(const Pair& p) {
  std::stringstream ss;
  ss << "{ x: " << unpack_x(p) << " y: " << (p.x << 16u) << " }";
  return ss.str(); 
}

int main() {
  wgpu::Device device = NativeUtils::CreateCppDawnDevice(nullptr);
  const int iterations = 5;
  
  SegmentedSort sorter;
  const uint32_t maxCount = 12000000;
  int maxNumSegments = ComputeUtil::div_up(maxCount, 100);

  wgpu::Buffer inputBuffer = utils::CreateBuffer(
    device, 
    maxCount * sizeof(Pair), 
    wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::CopySrc
  );

  wgpu::Buffer segmentsBuffer = utils::CreateBuffer(
    device, 
    maxNumSegments * sizeof(int), 
    wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst
  );

  sorter.Init(device, inputBuffer, maxCount, segmentsBuffer, maxNumSegments);
  for (uint32_t count = 1920; count < 8000000;  count += count / 10) {
    uint64_t gpu_time = 0;
    int numSegments = ComputeUtil::div_up(count, 100);

    for (uint32_t it = 0; it < iterations; it++)  {
      std::vector<Pair> vec = ComputeUtil::fill_random_pairs(0, INT16_MAX, count);
      std::vector<int> segments = ComputeUtil::fill_random_cpu(0, count - 1, numSegments, true);

      device.GetQueue().WriteBuffer(
       inputBuffer, 
       0, 
       vec.data(), 
       vec.size() * sizeof(int) * 2
      );
      device.GetQueue().WriteBuffer(
       segmentsBuffer, 
       0, 
       segments.data(), 
       segments.size() * sizeof(int)
      );

      ComputeUtil::BusyWaitDevice(device);
      auto t0 = high_resolution_clock::now();
      wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
      wgpu::ComputePassEncoder computePass = encoder.BeginComputePass();
      sorter.Sort(device, computePass, count, numSegments);
      computePass.End();
      auto commandBuffer = encoder.Finish();
      device.GetQueue().Submit(1, &commandBuffer);
      ComputeUtil::BusyWaitDevice(device);

      auto t1 = high_resolution_clock::now();
      gpu_time += duration_cast<nanoseconds>(t1-t0).count();

      auto cmp = [](const Pair& a, const Pair& b) -> bool {
          int ay = (a.x << 16);
          int by = (b.x << 16);
          if (ay < by) {
            return true;
          }

          if (ay == by) {
            return unpack_x(a) < unpack_x(b);
          }

          return false;
      };
      
      auto output = CopyReadBackBuffer<Pair>(device, inputBuffer, count * sizeof(Pair));

      std::vector<Pair> copy = vec;
      int cur = 0;
      for(int seg = 0; seg < segments.size(); ++seg) {
        int next = segments[seg];
        std::sort(copy.data() + cur, copy.data() + next, cmp);
        cur = next;
      }
      std::sort(copy.data() + cur, copy.data() + vec.size(), cmp);

      for(int i = 0; i < output.size(); i++) {
        if(copy[i].x != output[i].x) {
          std::cerr << "Faulty at count " << count << " " << i << ": " << output[i].x << "," << output[i].y << "  " << copy[i].x << "," << copy[i].y << std::endl;
          exit(1);
        } 
      }
    }
    std::cout << count << " " << (gpu_time / iterations) / 1000 << std::endl;
  }

  sorter.Dispose();
  inputBuffer.Destroy();
  segmentsBuffer.Destroy();
  return 0;
}
