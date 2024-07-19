#include <iostream>
#include <sstream>

#include "ComputeUtil.h"
#include "SegSort.h"
#include "Subgroups.h"
#include "wgpu/NativeUtils.h"

static std::unique_ptr<wgpu::Instance> instance;

static const wgpu::BufferUsage storageUsage = wgpu::BufferUsage::Storage;
static const wgpu::BufferUsage copyDstUsage = storageUsage | wgpu::BufferUsage::CopyDst;
static const wgpu::BufferUsage copySrcUsage = storageUsage | wgpu::BufferUsage::CopySrc;
static const wgpu::BufferUsage copyAllUsage = copySrcUsage | copyDstUsage;

uint32_t unpack_x(const uint2& p) { return p.x & 0xffffu; }

uint32_t unpack_y(const uint2& p) { return p.x >> 16u; }

struct QueryContainer {
    QueryContainer(const wgpu::Device& device, uint32_t numQueries) : numQueries{numQueries} {
        buffer = utils::CreateBuffer(
            device, numQueries * sizeof(uint64_t),
            wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::QueryResolve, "QuerySet");

        wgpu::QuerySetDescriptor querySetDescriptor;
        querySetDescriptor.count = numQueries;
        querySetDescriptor.label = "ComputeTime";
        querySetDescriptor.type = wgpu::QueryType::Timestamp;
        querySet = device.CreateQuerySet(&querySetDescriptor);

        gpu_times.resize(numQueries / 2u);
    }

    void Resolve(const wgpu::CommandEncoder& encoder) const {
        encoder.ResolveQuerySet(querySet, 0, numQueries, buffer, 0);
    }

    void Read(const wgpu::Device& device) {
        std::vector<uint64_t> queryData =
            ComputeUtil::CopyReadBackBuffer<uint64_t>(device, buffer, numQueries * sizeof(uint64_t));

        for (int i = 0; i < numQueries / 2; i++) {
            gpu_times[i] += queryData[i * 2 + 1] - queryData[i * 2];
        }
    }

    const std::vector<uint64_t>& GetTimings() const { return gpu_times; }

    uint64_t GetTotal() const {
        uint64_t total = 0u;
        for (int i = 0; i < numQueries / 2; i++) {
            total += gpu_times[i];
        }
        return total;
    }

    void Reset() {
        for (int i = 0; i < gpu_times.size(); i++) {
            gpu_times[i] = 0u;
        }
    }

    std::vector<uint64_t> gpu_times;
    const uint32_t numQueries;
    wgpu::Buffer buffer;
    wgpu::QuerySet querySet;
};

std::string PrintPos(const uint2& p) {
    std::stringstream ss;
    ss << "{ x: " << unpack_x(p) << " y: " << unpack_y(p) << " }";
    return ss.str();
}

void TestSubgroups(const std::unique_ptr<wgpu::Instance>& instance, const wgpu::Device& device) {
    SubgroupSort sorter;
    QueryContainer queryContainer(device, 2);
    uint32_t iterations = 10;

    uint32_t count = 1u << 27u;

    std::vector<int> data = ComputeUtil::fill_random_cpu(0, UINT32_MAX, count, false);
    wgpu::Buffer inputBuffer =
        utils::CreateBufferFromData(device, data.data(), data.size() * sizeof(uint32_t), copyAllUsage, "InputData");
    sorter.Init(device, inputBuffer, count);
    sorter.Upload(device, count);

    uint64_t total = 0u;
    for (int i = 0; i < iterations; i++) {
        std::vector<int> data = ComputeUtil::fill_random_cpu(0, UINT32_MAX, count, false);
        queryContainer.Reset();
        wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
        encoder.WriteBuffer(inputBuffer, 0, reinterpret_cast<const uint8_t*>(data.data()),
                            data.size() * sizeof(uint32_t));

        ComputeUtil::BusyWaitDevice(instance, device);
        sorter.Sort(encoder, queryContainer.querySet, count);
        queryContainer.Resolve(encoder);

        auto commandBuffer = encoder.Finish();
        device.GetQueue().Submit(1, &commandBuffer);
        ComputeUtil::BusyWaitDevice(instance, device);
        queryContainer.Read(device);
        total += queryContainer.GetTotal();
    }

    std::vector<uint32_t> output =
        ComputeUtil::CopyReadBackBuffer<uint32_t>(device, inputBuffer, count * sizeof(uint32_t));

    for (int i = 0; i < count; i += 32) {
        uint32_t last = 0u;
        for (int i = 0; i < 32; i++) {
            uint32_t v = output[i];
            if (v < last) {
                std::cerr << "Sort failed" << std::endl;
                exit(1);
            }
            // seen[v] = true;
            last = v;
        }
    }

    std::cout << "Total: " << total / static_cast<float>(1000 * 1000 * iterations) << " " << total << std::endl;
}

void TestSegsort(const std::unique_ptr<wgpu::Instance>& instance, const wgpu::Device& device) {
    const int iterations = 20;

    SegmentedSort sorter;
    const uint32_t maxCount = 12000000;
    int maxNumSegments = ComputeUtil::div_up(maxCount, 100);

    wgpu::Buffer inputBuffer = utils::CreateBuffer(
        device, maxCount * sizeof(int2),
        wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::CopySrc, "InputBuffer");

    wgpu::Buffer segmentsBuffer =
        utils::CreateBuffer(device, maxNumSegments * sizeof(int),
                            wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst, "SegmentsBuffer");

    const uint32_t numQueries = 8;
    QueryContainer queryContainer(device, numQueries);

    sorter.Init(device, inputBuffer, maxCount, segmentsBuffer, maxNumSegments);
    for (uint32_t count = 2000000; count <= 2000000; count += count / 10) {
        uint64_t cpu_time = 0;
        // std::array<uint64_t, numQueries / 2> gpu_times;
        // for (int i = 0; i < gpu_times.size(); i++) {
        //     gpu_times[i] = 0u;
        // }

        queryContainer.Reset();

        int numSegments = ComputeUtil::div_up(count, 100);

        for (uint32_t it = 0; it < iterations; it++) {
            std::vector<uint2> vec = ComputeUtil::fill_random_pairs(0, UINT32_MAX, count);
            std::vector<int> segments = ComputeUtil::fill_random_cpu(0, count - 1, numSegments, true);

            device.GetQueue().WriteBuffer(inputBuffer, 0, vec.data(), vec.size() * sizeof(uint2));
            device.GetQueue().WriteBuffer(segmentsBuffer, 0, segments.data(), segments.size() * sizeof(int));
            sorter.Upload(device, count, numSegments);

            ComputeUtil::BusyWaitDevice(instance, device);

            wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
            sorter.Sort(encoder, queryContainer.querySet, count, numSegments);
            queryContainer.Resolve(encoder);
            auto commandBuffer = encoder.Finish();

            auto t0 = high_resolution_clock::now();

            device.GetQueue().Submit(1, &commandBuffer);
            ComputeUtil::BusyWaitDevice(instance, device);
            queryContainer.Read(device);

            auto t1 = high_resolution_clock::now();

            cpu_time += duration_cast<nanoseconds>(t1 - t0).count();

            auto cmp = [](const uint2& a, const uint2& b) -> bool { return a.x < b.x; };

            std::vector<uint2> output =
                ComputeUtil::CopyReadBackBuffer<uint2>(device, inputBuffer, count * sizeof(int2));

            std::vector<uint2> copy = vec;
            int cur = 0;
            for (int seg = 0; seg < segments.size(); seg++) {
                int next = segments[seg];
                std::sort(copy.data() + cur, copy.data() + next, cmp);
                cur = next;
            }
            std::sort(copy.data() + cur, copy.data() + vec.size(), cmp);

            for (int i = 0; i < output.size(); i++) {
                if (copy[i].x != output[i].x) {
                    std::cerr << "Faulty at count " << count << " - i:" << i << ": " << PrintPos(output[i])
                              << " expected: " << PrintPos(copy[i]) << std::endl;
                    exit(1);
                }
            }
        }

        std::cout << count << " " << (cpu_time / iterations) / 1000 << std::endl;
        uint64_t tot_gpu = 0u;

        auto& gpu_times = queryContainer.GetTimings();
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

    std::vector<const char*> enableToggleNames = {"allow_unsafe_apis"};
    std::vector<const char*> disabledToggleNames = {};

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

    TestSubgroups(instance, device);
    device.Destroy();
}