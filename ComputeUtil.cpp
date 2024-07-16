#include "ComputeUtil.h"

// #include "src/util/FileUtil.h"
// #include "src/2d/renderer/HotReloadShader.h"
#include <thread>

namespace ComputeUtil  {
    void HandleError(WGPUErrorType type, char const * message, void * userdata) {
        std::cerr << "Error: " << message << std::endl;
        *static_cast<bool*>(userdata) = true; 
    }

  wgpu::ComputePipeline CreatePipeline(
      const wgpu::Device& device, 
      const wgpu::BindGroupLayout& bgl, 
      const std::string& shader,
      const char* label) {
    wgpu::ShaderModule shaderModule = utils::CreateShaderModule(device, shader.c_str(), label);
    wgpu::PipelineLayout pl = utils::MakeBasicPipelineLayout(device, &bgl);
    wgpu::ComputePipelineDescriptor csDesc;
    csDesc.layout = pl;
    csDesc.compute.module = shaderModule;
    csDesc.compute.entryPoint = "main";
    csDesc.label = label;
    // Needed for emscripten
    csDesc.compute.constantCount = 0;
    return device.CreateComputePipeline(&csDesc);
  }

    inline std::mt19937& get_mt19937() {
        static std::mt19937 mt19937;
        return mt19937;
    }

    template <typename T>
    void PrintRange(const std::vector<T>& data, int start, int end, uint32_t divisor) {
        for(int i = start; i < end; i++){
            std::cout << data[i] << " ";
            if((i+1) % divisor == 0) std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    std::vector<uint2> fill_random_pairs(int a, int b, size_t count) {
        std::uniform_int_distribution<int> d(a, b);
        std::vector<uint2> data(count);

        for (uint2& i : data) {
            i.x = d(get_mt19937());
            i.y = d(get_mt19937());
        }
            
        return data;
    }

    std::vector<int> fill_random_cpu(int a, int b, size_t count, bool sorted) {
        std::uniform_int_distribution<int> d(a, b);
        std::vector<int> data(count);

        for(int& i : data)
            i = d(get_mt19937());
        
        if (sorted) {
            std::sort(data.begin(), data.end());
        }
        return data;
    }
    // Count leading zeros
    int clz(int x) {
        for(int i = 31; i >= 0; --i)
            if((1<< i) & x) return 31 - i;
        return 32;
    }

    constexpr bool is_pow2(int x) {
        return 0 == (x & (x - 1));
    }

    int32_t div_up(int32_t x, int32_t y) {
        return (x + y - 1) / y;
    }

    int find_log2(int x, bool round_up) {
        int a = 31 - clz(x);
        if (round_up) { 
            a += !is_pow2(x); 
        }
        return a;
    } 

    void PrintGPUBuffer(const wgpu::Device& device, const wgpu::Buffer& buffer, uint32_t byteSize, uint32_t newLineCount) {
        auto pvec = CopyReadBackBuffer<int>(device, buffer, byteSize);
        PrintRange(pvec, 0, pvec.size(), newLineCount);
    }

    void BusyWaitDevice(const wgpu::Device& device) {
        bool done = false;
        device.GetQueue().OnSubmittedWorkDone([](WGPUQueueWorkDoneStatus status, void * userdata) -> void {
            *static_cast<bool*>(userdata) = true;
        }, &done);

        while (!done) {
            #ifndef __EMSCRIPTEN__
            device.Tick();
            #endif
        }
    }
}


// template <typename T, typename C>
// std::vector<T> cpuSegsort(const std::vector<T>& data, const std::vector<int>& segments, C cmp) {
//     std::vector<T> copy = data;
//     int cur = 0;
//     for(int seg = 0; seg < segments.size(); ++seg) {
//         int next = segments[seg];
//         std::sort(copy.data() + cur, copy.data() + next, cmp);
//         cur = next;
//     }
//     std::sort(copy.data() + cur, copy.data() + data.size(), cmp);
//     return copy;
// }