// Copyright 2017 The Dawn Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "WGPUHelpers.h"

#include <cstring>
#include <iomanip>
#include <limits>
#include <mutex>
#include <sstream>
#include <thread>

#if TINT_BUILD_SPV_READER
#include "spirv-tools/optimizer.hpp"
#endif

namespace {
std::array<float, 12> kYuvToRGBMatrixBT709 = {1.164384f,  0.0f,      1.792741f, -0.972945f, 1.164384f, -0.213249f,
                                              -0.532909f, 0.301483f, 1.164384f, 2.112402f,  0.0f,      -1.133402f};
std::array<float, 9> kGamutConversionMatrixBT709ToSrgb = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};
std::array<float, 7> kGammaDecodeBT709 = {2.2, 1.0 / 1.099, 0.099 / 1.099, 1 / 4.5, 0.081, 0.0, 0.0};
std::array<float, 7> kGammaEncodeSrgb = {1 / 2.4, 1.137119, 0.0, 12.92, 0.0031308, -0.055, 0.0};
}  // namespace

namespace utils {
#if TINT_BUILD_SPV_READER
wgpu::ShaderModule CreateShaderModuleFromASM(const wgpu::Device& device, const char* source,
                                             wgpu::DawnShaderModuleSPIRVOptionsDescriptor* spirv_options) {
    // Use SPIRV-Tools's C API to assemble the SPIR-V assembly text to binary. Because the types
    // aren't RAII, we don't return directly on success and instead always go through the code
    // path that destroys the SPIRV-Tools objects.
    wgpu::ShaderModule result = nullptr;

    spv_context context = spvContextCreate(SPV_ENV_UNIVERSAL_1_3);
    DAWN_ASSERT(context != nullptr);

    spv_binary spirv = nullptr;
    spv_diagnostic diagnostic = nullptr;
    if (spvTextToBinary(context, source, strlen(source), &spirv, &diagnostic) == SPV_SUCCESS) {
        DAWN_ASSERT(spirv != nullptr);
        DAWN_ASSERT(spirv->wordCount <= std::numeric_limits<uint32_t>::max());

        wgpu::ShaderModuleSPIRVDescriptor spirvDesc;
        spirvDesc.codeSize = static_cast<uint32_t>(spirv->wordCount);
        spirvDesc.code = spirv->code;
        spirvDesc.nextInChain = spirv_options;

        wgpu::ShaderModuleDescriptor descriptor;
        descriptor.nextInChain = &spirvDesc;
        result = device.CreateShaderModule(&descriptor);
    } else {
        DAWN_ASSERT(diagnostic != nullptr);
        dawn::WarningLog() << "CreateShaderModuleFromASM SPIRV assembly error:" << diagnostic->position.line + 1 << ":"
                           << diagnostic->position.column + 1 << ": " << diagnostic->error;
    }

    spvDiagnosticDestroy(diagnostic);
    spvBinaryDestroy(spirv);
    spvContextDestroy(context);

    return result;
}
#endif

wgpu::ShaderModule CreateShaderModule(const wgpu::Device& device, const char* source, const char* label) {
    wgpu::ShaderModuleWGSLDescriptor wgslDesc;
    wgslDesc.code = source;
    wgpu::ShaderModuleDescriptor descriptor;
    descriptor.nextInChain = &wgslDesc;
    descriptor.label = label;
    return device.CreateShaderModule(&descriptor);
}

wgpu::Buffer CreateBufferFromData(const wgpu::Device& device, const void* data, uint64_t size, wgpu::BufferUsage usage,
                                  const char* label) {
    wgpu::BufferDescriptor descriptor;
    descriptor.size = size;
    descriptor.usage = usage | wgpu::BufferUsage::CopyDst;
    descriptor.label = label;
    wgpu::Buffer buffer = device.CreateBuffer(&descriptor);

    device.GetQueue().WriteBuffer(buffer, 0, data, size);
    return buffer;
}

wgpu::Buffer CreateBuffer(const wgpu::Device& device, uint64_t size, wgpu::BufferUsage usage, const char* label) {
    wgpu::BufferDescriptor descriptor;
    descriptor.size = size;
    descriptor.usage = usage | wgpu::BufferUsage::CopyDst;
    descriptor.label = label;
    return device.CreateBuffer(&descriptor);
}

void BusyWaitDevice(const wgpu::Device& device) {
    bool done = false;
    device.GetQueue().OnSubmittedWorkDone(
        [](WGPUQueueWorkDoneStatus status, void* userdata) -> void { *(static_cast<bool*>(userdata)) = true; }, &done);

    while (!done) {
        #ifndef __EMSCRIPTEN__
        device.Tick();
        #endif
        std::this_thread::sleep_for(std::chrono::microseconds{1});
    }
}


wgpu::PipelineLayout MakeBasicPipelineLayout(const wgpu::Device& device, const wgpu::BindGroupLayout* bindGroupLayout,
                                             uint32_t bindGroupLayerCount, const wgpu::ChainedStruct* nextInChain) {
    wgpu::PipelineLayoutDescriptor descriptor;
    if (bindGroupLayout != nullptr) {
        descriptor.bindGroupLayoutCount = bindGroupLayerCount;
        descriptor.bindGroupLayouts = bindGroupLayout;
    } else {
        descriptor.bindGroupLayoutCount = 0;
        descriptor.bindGroupLayouts = nullptr;
    }

    if (nextInChain != nullptr) {
        descriptor.nextInChain = nextInChain;
    }

    return device.CreatePipelineLayout(&descriptor);
}

wgpu::PipelineLayout MakePipelineLayout(const wgpu::Device& device, std::vector<wgpu::BindGroupLayout> bgls) {
    wgpu::PipelineLayoutDescriptor descriptor;
    descriptor.bindGroupLayoutCount = uint32_t(bgls.size());
    descriptor.bindGroupLayouts = bgls.data();
    return device.CreatePipelineLayout(&descriptor);
}

wgpu::BindGroupLayout MakeBindGroupLayout(const wgpu::Device& device, const char* label,
                                          std::vector<BindingLayoutEntryInitializationHelper> entriesInitializer,
                                          const wgpu::ChainedStruct* nextInChain) {
    std::vector<wgpu::BindGroupLayoutEntry> entries;

    for (const BindingLayoutEntryInitializationHelper& entry : entriesInitializer) {
        entries.push_back(entry);
    }

    wgpu::BindGroupLayoutDescriptor descriptor;
    descriptor.entryCount = static_cast<uint32_t>(entries.size());
    descriptor.entries = entries.data();
    descriptor.label = label;
    return device.CreateBindGroupLayout(&descriptor);
}

wgpu::BindGroupLayout MakeBindGroupLayout(
    const wgpu::Device& device, const char* label,
    std::initializer_list<BindingLayoutEntryInitializationHelper> entriesInitializer,
    const wgpu::ChainedStruct* nextInChain) {
    std::vector<wgpu::BindGroupLayoutEntry> entries;
    for (const BindingLayoutEntryInitializationHelper& entry : entriesInitializer) {
        wgpu::BindGroupLayoutEntry l;
        entries.push_back(entry);
    }

    wgpu::BindGroupLayoutDescriptor descriptor;
    descriptor.entryCount = entries.size();
    descriptor.entries = entries.data();
    descriptor.label = label;
    if (nextInChain != nullptr) {
        descriptor.nextInChain = nextInChain;
    }
    return device.CreateBindGroupLayout(&descriptor);
}

BindingLayoutEntryInitializationHelper::BindingLayoutEntryInitializationHelper(uint32_t entryBinding,
                                                                               wgpu::ShaderStage entryVisibility,
                                                                               wgpu::BufferBindingType bufferType,
                                                                               bool bufferHasDynamicOffset,
                                                                               uint64_t bufferMinBindingSize) {
    binding = entryBinding;
    visibility = entryVisibility;
    buffer.type = bufferType;
    buffer.hasDynamicOffset = bufferHasDynamicOffset;
    buffer.minBindingSize = bufferMinBindingSize;
}

BindingLayoutEntryInitializationHelper::BindingLayoutEntryInitializationHelper(uint32_t entryBinding,
                                                                               wgpu::ShaderStage entryVisibility,
                                                                               wgpu::SamplerBindingType samplerType) {
    binding = entryBinding;
    visibility = entryVisibility;
    sampler.type = samplerType;
}

BindingLayoutEntryInitializationHelper::BindingLayoutEntryInitializationHelper(
    uint32_t entryBinding, wgpu::ShaderStage entryVisibility, wgpu::TextureSampleType textureSampleType,
    wgpu::TextureViewDimension textureViewDimension, bool textureMultisampled) {
    binding = entryBinding;
    visibility = entryVisibility;
    texture.sampleType = textureSampleType;
    texture.viewDimension = textureViewDimension;
    texture.multisampled = textureMultisampled;
}

BindingLayoutEntryInitializationHelper::BindingLayoutEntryInitializationHelper(
    uint32_t entryBinding, wgpu::ShaderStage entryVisibility, wgpu::StorageTextureAccess storageTextureAccess,
    wgpu::TextureFormat format, wgpu::TextureViewDimension textureViewDimension) {
    binding = entryBinding;
    visibility = entryVisibility;
    storageTexture.access = storageTextureAccess;
    storageTexture.format = format;
    storageTexture.viewDimension = textureViewDimension;
}

// ExternalTextureBindingLayout never contains data, so just make one that can be reused instead
// of declaring a new one every time it's needed.
#ifndef __EMSCRIPTEN__
wgpu::ExternalTextureBindingLayout kExternalTextureBindingLayout = {};

BindingLayoutEntryInitializationHelper::BindingLayoutEntryInitializationHelper(
    uint32_t entryBinding, wgpu::ShaderStage entryVisibility, wgpu::ExternalTextureBindingLayout* bindingLayout) {
    binding = entryBinding;
    visibility = entryVisibility;
    nextInChain = bindingLayout;
}
#endif

BindingLayoutEntryInitializationHelper::BindingLayoutEntryInitializationHelper(const wgpu::BindGroupLayoutEntry& entry)
    : wgpu::BindGroupLayoutEntry(entry) {}

BindingInitializationHelper::BindingInitializationHelper(uint32_t binding, const wgpu::Sampler& sampler)
    : binding(binding), sampler(sampler) {}

BindingInitializationHelper::BindingInitializationHelper(uint32_t binding, const wgpu::TextureView& textureView)
    : binding(binding), textureView(textureView) {}

#ifndef __EMSCRIPTEN__
BindingInitializationHelper::BindingInitializationHelper(uint32_t binding, const wgpu::ExternalTexture& externalTexture)
    : binding(binding) {
    externalTextureBindingEntry.externalTexture = externalTexture;
}
#endif

BindingInitializationHelper::BindingInitializationHelper(uint32_t binding, const wgpu::Buffer& buffer, uint64_t offset,
                                                         uint64_t size)
    : binding(binding), buffer(buffer), offset(offset), size(size) {}

BindingInitializationHelper::BindingInitializationHelper(const BindingInitializationHelper&) = default;

BindingInitializationHelper::~BindingInitializationHelper() = default;

wgpu::BindGroupEntry BindingInitializationHelper::GetAsBinding() const {
    wgpu::BindGroupEntry result;

    result.binding = binding;
    result.sampler = sampler;
    result.textureView = textureView;
    result.buffer = buffer;
    result.offset = offset;
    result.size = size;
    #ifndef __EMSCRIPTEN__
    if (externalTextureBindingEntry.externalTexture != nullptr) {
        result.nextInChain = &externalTextureBindingEntry;
    }
    #endif

    return result;
}

const wgpu::BindGroup MakeBindGroup(const wgpu::Device& device, const wgpu::BindGroupLayout& layout,
                              const std::vector<BindingInitializationHelper>& entriesInitializer) {
    std::vector<wgpu::BindGroupEntry> entries;

    for (const BindingInitializationHelper& helper : entriesInitializer) {
        entries.push_back(helper.GetAsBinding());
    }

    wgpu::BindGroupDescriptor descriptor;
    descriptor.layout = layout;
    descriptor.entryCount = entries.size();
    descriptor.entries = entries.data();

    return device.CreateBindGroup(&descriptor);
}

const wgpu::BindGroup MakeBindGroup(const wgpu::Device& device, const wgpu::BindGroupLayout& layout,
                              std::initializer_list<BindingInitializationHelper> entriesInitializer) {
    std::vector<wgpu::BindGroupEntry> entries;
    for (const BindingInitializationHelper& helper : entriesInitializer) {
        entries.push_back(helper.GetAsBinding());
    }

    wgpu::BindGroupDescriptor descriptor;
    descriptor.layout = layout;
    descriptor.entryCount = static_cast<uint32_t>(entries.size());
    descriptor.entries = entries.data();

    return device.CreateBindGroup(&descriptor);
}

ColorSpaceConversionInfo GetYUVBT709ToRGBSRGBColorSpaceConversionInfo() {
    ColorSpaceConversionInfo info;
    info.yuvToRgbConversionMatrix = kYuvToRGBMatrixBT709;
    info.gamutConversionMatrix = kGamutConversionMatrixBT709ToSrgb;
    info.srcTransferFunctionParameters = kGammaDecodeBT709;
    info.dstTransferFunctionParameters = kGammaEncodeSrgb;

    return info;
}

}  // namespace utils
