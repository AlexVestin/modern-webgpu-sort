#pragma once

#include <iostream>
#include <string>
#include <fstream>

#include "Renderer.h"

#include "defs.h"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "stb_image.h"
#pragma GCC diagnostic pop

#include "RendererUtil.h"

#include "path/VPoint.h"
#include "../wgpu/NativeUtils.h"
#include "../ComputeUtil.h"
#include "../wgpu/ComboRenderPipelineDescriptor.h"
#include "../wgpu/Blends.h"
#include "RendererUtil.h"

static std::unique_ptr<wgpu::Instance> instance;

static std::array<uint8_t, IMAGE_WIDTH * IMAGE_HEIGHT> outAtlas = {};

const wgpu::TextureFormat atlasFormat = wgpu::TextureFormat::R32Float;


std::string ReadTextFile(const std::string& path) {
    std::ifstream t(path.c_str());
    
    if (t.fail()) {
        std::cerr << "Failed to find file" << std::endl;
        exit(1);
    }
    std::stringstream buffer;
    buffer << t.rdbuf();
    return buffer.str();
}

Renderer::Renderer(uint32_t atlasWidth, uint32_t atlasHeight): atlasWidth{atlasWidth}, atlasHeight{atlasHeight} {
    InitDevice();
    
    drawBindGroupLayout = utils::MakeBindGroupLayout(device, "DrawBindGroupLayout", {
        {0, wgpu::ShaderStage::Vertex | wgpu::ShaderStage::Fragment, wgpu::BufferBindingType::ReadOnlyStorage},
        {1, wgpu::ShaderStage::Vertex | wgpu::ShaderStage::Fragment, wgpu::BufferBindingType::ReadOnlyStorage},
        {2, wgpu::ShaderStage::Vertex | wgpu::ShaderStage::Fragment, wgpu::BufferBindingType::ReadOnlyStorage},
        {3, wgpu::ShaderStage::Vertex | wgpu::ShaderStage::Fragment, wgpu::BufferBindingType::ReadOnlyStorage},
        {4, wgpu::ShaderStage::Vertex | wgpu::ShaderStage::Fragment, wgpu::BufferBindingType::ReadOnlyStorage},
    });

    // Create atlas texture
    atlasTexture = CreateTexture(atlasFormat);
    atlasTextureView = atlasTexture.CreateView();

    // Create empty buffers
    pathInfoBuffer  = utils::CreateBuffer(device, 4, copyDstUsage, "ColorBuffer");
    lineIndexBuffer  = utils::CreateBuffer(device, 4, copyDstUsage, "LineIndexBuffer");
    flatLinePointBuffer  = utils::CreateBuffer(device, 4, copyDstUsage, "FlatLinePointBuffer");
    drawSpansBuffer  = utils::CreateBuffer(device, 4, copyDstUsage, "DrawSpansBuffer");
    atlasIndicesBuffer  = utils::CreateBuffer(device, 4, copyDstUsage, "AtlasIndicesBuffer");

    textureDataBuffer = utils::CreateBuffer(device, IMAGE_WIDTH * IMAGE_HEIGHT * sizeof(float), copySrcUsage, "CopyTextureBuffer");

    std::string atlasShader = ReadTextFile("vector/shaders/atlas.wgsl");
    wgpu::ShaderModule shaderModule = utils::CreateShaderModule(device, atlasShader.c_str(), "AtlasShader");
    pipeline = CreateRenderPipeline(device,
                                    {.vertModule = shaderModule,
                                    .fragModule = shaderModule,
                                    .targetFormat = atlasFormat,
                                    .blendState = &lyra::blend::Additive,
                                    .bindGroupLayouts = {drawBindGroupLayout}},
                                    "AtlasPipeline");
    }

void Renderer::InitDevice() {
    dawnProcSetProcs(&dawn::native::GetProcs());

    std::vector<const char*> enableToggleNames = {"allow_unsafe_apis", "dump_shaders"};
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
    device = NativeUtils::SetupDevice(instance, adapter);
}


wgpu::Texture Renderer::CreateTexture(const wgpu::TextureFormat format) const {
    wgpu::TextureDescriptor descriptor;
    descriptor.size.width = 1920;
    descriptor.size.height = 1080;
    // descriptor.size.depthOrArrayLayers = ;
    descriptor.dimension = wgpu::TextureDimension::e2D;
    descriptor.sampleCount = 1;
    descriptor.format = format;
    descriptor.mipLevelCount = 1;
    descriptor.usage = wgpu::TextureUsage::CopySrc | wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::RenderAttachment;
    return device.CreateTexture(&descriptor);
}

void Renderer::Render(uint32_t atlasIndices, uint32_t drawSpans) const {
    wgpu::CommandEncoder encoder = device.CreateCommandEncoder();

    utils::ComboRenderPassDescriptor atlasPassDescriptor({atlasTextureView});
    atlasPassDescriptor.cColorAttachments[0].loadOp = wgpu::LoadOp::Load;
    wgpu::RenderPassEncoder atlasPass = encoder.BeginRenderPass(&atlasPassDescriptor);
    atlasPass.SetBindGroup(0, bindGroup);
    atlasPass.SetPipeline(pipeline);
    atlasPass.Draw(atlasIndices * 6u);
    atlasPass.End();

    wgpu::CommandBuffer commandBuffer = encoder.Finish();
    device.GetQueue().Submit(1, &commandBuffer);

    WriteAtlasTexture();
}

void Renderer::WriteAtlasTexture() const {
    uint32_t width = 1920;
    uint32_t height = 1080;
    wgpu::Buffer buf = ReadBackTexture(device, atlasTexture, width, height, 4);
    const float* data = static_cast<const float*>(buf.GetConstMappedRange());

    for (int i = 0; i < width * height; i++) {
        float area = data[i];
        // float a = std::min(std::abs(area - 2.0f * std::round(0.5f * area)), 1.0f);
        outAtlas[i] = static_cast<uint8_t>(std::clamp(area, 0.0f, 1.0f) * 255.0f); 
    }
    uint32_t bpr = width;
    stbi_write_png("written.png", width, height, 1, static_cast<const void*>(outAtlas.data()), bpr);
    buf.Destroy();
}

void Renderer::CreateBindGroup() {
    bindGroup = utils::MakeBindGroup(device, drawBindGroupLayout, {
        {0, pathInfoBuffer},
        {1, lineIndexBuffer},
        {2, flatLinePointBuffer},
        {3, drawSpansBuffer},
        {4, atlasIndicesBuffer}
    });
}


void Renderer::Upload(
    const std::vector<uint32_t>& colors,
    const std::vector<VPoint>& flatPoints,
    const std::vector<uint32_t>& indices,
    const std::vector<DrawSpan>& drawSpans,
    const std::vector<uint32_t>& atlasIndices) {
    uploadAmount = 0u;
    
    CreateOrUploadBuffer(device, &pathInfoBuffer, colors, "ColorBuffer");
    CreateOrUploadBuffer(device, &lineIndexBuffer, indices, "LineIndexBuffer");
    CreateOrUploadBuffer(device, &flatLinePointBuffer, flatPoints, "FlatPointBuffer");
    CreateOrUploadBuffer(device, &drawSpansBuffer, drawSpans, "DrawSpansBuffer");
    CreateOrUploadBuffer(device, &atlasIndicesBuffer, atlasIndices, "AtlasIndicesBuffer");

    if (needsRecreateBindGroup) {
        CreateBindGroup();
    }

    // std::cout << "Uploaded: " << uploadAmount << std::endl;
}

void Renderer::Dispose() {
    pathInfoBuffer.Destroy();
    lineIndexBuffer.Destroy();
    flatLinePointBuffer.Destroy();
    drawSpansBuffer.Destroy();
    atlasIndicesBuffer.Destroy();
    device.Destroy();
}
