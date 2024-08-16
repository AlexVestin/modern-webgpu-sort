#pragma once

#include <vector>
#include <thread>
#include <iostream>
#include <webgpu/webgpu_cpp.h>


#include "../wgpu/WGPUHelpers.h"
#include "../wgpu/ComboRenderPipelineDescriptor.h"
#include "../wgpu/Blends.h"



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

    fprintf(stderr, "Failed to read back buffer, with status: %d\n", static_cast<int>(readStatus));
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



struct VertexBufferLayout {
    wgpu::VertexStepMode stepMode = wgpu::VertexStepMode::Vertex;
    uint32_t stride = 0u;
    std::vector<wgpu::VertexAttribute> attributes = {};
};

struct RenderPipelineCreationOptions {
    wgpu::ShaderModule vertModule;
    wgpu::ShaderModule fragModule;
    const wgpu::BlendState* blendState = &lyra::blend::OneMinusSrcAlpha;
    wgpu::MultisampleState multisampleState = {.count = 1, .mask = 0xFFFFFFFF, .alphaToCoverageEnabled = false};
    std::string vertexEntryPoint = "vert_main";
    std::string fragmentEntryPoint = "frag_main";
    wgpu::TextureFormat targetFormat = wgpu::TextureFormat::RGBA8Unorm;
    wgpu::PrimitiveTopology topology = wgpu::PrimitiveTopology::TriangleList;
    std::vector<wgpu::ConstantEntry> constants = {};
    wgpu::ColorWriteMask targetColorWriteMask = wgpu::ColorWriteMask::All;
    std::vector<VertexBufferLayout> vertexBufferLayouts = {};
    const std::vector<wgpu::BindGroupLayout> bindGroupLayouts;
    const wgpu::ChainedStruct* nextInChain = nullptr;
};

wgpu::RenderPipeline CreateRenderPipeline(const wgpu::Device& device, const RenderPipelineCreationOptions& options,
                                          const std::string& label) {
    utils::ComboRenderPipelineDescriptor descriptor;
    descriptor.label = label.c_str();

    if (!options.bindGroupLayouts.empty()) {
        descriptor.layout = utils::MakeBasicPipelineLayout(device, options.bindGroupLayouts.data(),
                                                           options.bindGroupLayouts.size(), options.nextInChain);
    }

    descriptor.vertex.module = options.vertModule;
    descriptor.vertex.entryPoint = options.vertexEntryPoint.c_str();
    descriptor.vertex.constantCount = options.constants.size();
    descriptor.vertex.constants = options.constants.data();

    descriptor.vertex.bufferCount = options.vertexBufferLayouts.size();

    for (int i = 0; i < options.vertexBufferLayouts.size(); i++) {
        auto& layout = options.vertexBufferLayouts[i];
        descriptor.cBuffers[i].arrayStride = layout.stride;
        descriptor.cBuffers[i].attributeCount = layout.attributes.size();
        descriptor.cBuffers[i].attributes = layout.attributes.data();
        descriptor.cBuffers[i].stepMode = layout.stepMode;
    }

    descriptor.primitive.topology = options.topology;

    // Fragment info
    descriptor.cFragment.module = options.fragModule;
    descriptor.cFragment.entryPoint = options.fragmentEntryPoint.c_str();
    descriptor.cFragment.constantCount = options.constants.size();
    descriptor.cFragment.constants = options.constants.data();

    // Target information
    descriptor.cFragment.targetCount = 1;
    
    descriptor.cTargets[0].blend = options.blendState;
    descriptor.cTargets[0].nextInChain = nullptr;
    descriptor.cTargets[0].format = options.targetFormat;
    descriptor.cTargets[0].writeMask = options.targetColorWriteMask;

    return device.CreateRenderPipeline(&descriptor);
}


wgpu::Buffer ReadBackTexture(const wgpu::Device& device, const wgpu::Texture& texture, uint32_t width, uint32_t height,
                             uint32_t channels) {
    uint32_t bpr = (((width * channels) + 255) / 256) * 256;

    std::cout << "bpr: " << bpr << " " << width * 4 << std::endl;
    uint32_t byteSize = bpr * height;
    wgpu::BufferDescriptor descriptor;
    descriptor.size = byteSize;
    descriptor.usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst;
    wgpu::Buffer readBuffer = device.CreateBuffer(&descriptor);

    wgpu::ImageCopyBuffer imageCopyBuffer = utils::CreateImageCopyBuffer(readBuffer, 0, bpr, height);
    wgpu::ImageCopyTexture imageCopyTexture = utils::CreateImageCopyTexture(texture, 0, {0, 0, 0});
    wgpu::Extent3D copySize = {width, height, 1};

    wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
    encoder.CopyTextureToBuffer(&imageCopyTexture, &imageCopyBuffer, &copySize);
    wgpu::CommandBuffer commands = encoder.Finish();
    device.GetQueue().Submit(1, &commands);

    bool done = false;
    readBuffer.MapAsync(
        wgpu::MapMode::Read, 0, byteSize,
        [](WGPUBufferMapAsyncStatus status, void* userdata) { *static_cast<bool*>(userdata) = true; }, &done);

    while (!done) {
#ifndef __EMSCRIPTEN__
        device.Tick();
        std::this_thread::sleep_for(std::chrono::milliseconds{1});
#endif
    }

    return readBuffer;
}

void WriteTextureToImage(const char* filename, const wgpu::Device& device, const wgpu::Texture& texture, uint32_t width,
                         uint32_t height, uint32_t channels) {
    const wgpu::Buffer& buffer = ReadBackTexture(device, texture, width, height, channels);
    const uint8_t* data = static_cast<const uint8_t*>(buffer.GetConstMappedRange());
    uint32_t bpr = (((width * channels) + 255) / 256) * 256;
    // stbi_write_png("written.png", bpr / 4, height, 4, static_cast<const void*>(data), bpr);
    buffer.Destroy();
}