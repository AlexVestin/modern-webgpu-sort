#include "SegSort.h"

#include <iostream>
#include "wgpu/NativeUtils.h"

static std::unique_ptr<wgpu::Instance> instance;

static const wgpu::BufferUsage storageUsage = wgpu::BufferUsage::Storage;
static const wgpu::BufferUsage copyDstUsage = storageUsage | wgpu::BufferUsage::CopyDst;
static const wgpu::BufferUsage copySrcUsage = storageUsage | wgpu::BufferUsage::CopySrc;
static const wgpu::BufferUsage copyAllUsage = copySrcUsage | copyDstUsage;

int main() {
    dawnProcSetProcs(&dawn::native::GetProcs());

    std::vector<const char*> enableToggleNames = { "allow_unsafe_apis", "disable_robustness" };
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

    uint32_t inputSize = 16u;
    uint32_t segmentSize = 16u;

    std::vector<uint2> data(inputSize);
    std::vector<uint32_t> segments(segmentSize);

    wgpu::Buffer segmentBuffer = utils::CreateBufferFromData(device, segments.data(), segments.size() * sizeof(uint2), copyDstUsage,  "SegmentsBuffer");
    wgpu::Buffer inputBuffer = utils::CreateBufferFromData(device, data.data(), data.size() * sizeof(uint32_t), copyDstUsage,  "InputBuffer");
    
    SegmentedSort sorter;
    sorter.Init(device, inputBuffer, inputSize, segmentBuffer, segmentSize);
}