#include <iostream>
#include <string>
#include <fstream>
#include <sstream>

#include "Renderer.h"

#include "defs.h"

#pragma GCC diagnostic push
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


struct Uniforms {
    uint2 viewSize;
    uint32_t tileSize;
    uint32_t padding;
};

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

    auto doubleStage = wgpu::ShaderStage::Vertex | wgpu::ShaderStage::Fragment;
    drawBindGroupLayout = utils::MakeBindGroupLayout(device, "DrawBindGroupLayout", {
        {0, doubleStage, wgpu::BufferBindingType::ReadOnlyStorage},
        {1, doubleStage, wgpu::BufferBindingType::ReadOnlyStorage},
        {2, doubleStage, wgpu::BufferBindingType::ReadOnlyStorage},
        {3, doubleStage, wgpu::BufferBindingType::ReadOnlyStorage},
        {4, doubleStage, wgpu::BufferBindingType::ReadOnlyStorage},
        {5, doubleStage, wgpu::BufferBindingType::Uniform},
    });

    atlasBindGroupLayout = utils::MakeBindGroupLayout(device, "AtlasBindGroupLayout", {
        {0, wgpu::ShaderStage::Fragment, wgpu::TextureSampleType::UnfilterableFloat},
        {1, wgpu::ShaderStage::Fragment, wgpu::SamplerBindingType::NonFiltering}
    });

    // Create atlas texture
    atlasTexture = CreateTexture(atlasFormat);
    atlasTextureView = atlasTexture.CreateView();

    drawTexture = CreateTexture(wgpu::TextureFormat::RGBA8Unorm);
    drawTextureView = drawTexture.CreateView();

    // Create empty buffers
    pathInfoBuffer  = utils::CreateBuffer(device, 4, copyDstUsage, "ColorBuffer");
    lineIndexBuffer  = utils::CreateBuffer(device, 4, copyDstUsage, "LineIndexBuffer");
    flatLinePointBuffer  = utils::CreateBuffer(device, 4, copyDstUsage, "FlatLinePointBuffer");
    drawSpansBuffer  = utils::CreateBuffer(device, 4, copyDstUsage, "DrawSpansBuffer");
    atlasIndicesBuffer  = utils::CreateBuffer(device, 4, copyDstUsage, "AtlasIndicesBuffer");

    textureDataBuffer = utils::CreateBuffer(device, IMAGE_WIDTH * IMAGE_HEIGHT * sizeof(float), copySrcUsage, "CopyTextureBuffer");

    std::string atlasShader = ReadTextFile("vector/shaders/atlas.wgsl");
    wgpu::ShaderModule atlasShaderModule = utils::CreateShaderModule(device, atlasShader.c_str(), "AtlasShader");
    atlasPipeline = CreateRenderPipeline(device,
                                    {.vertModule = atlasShaderModule,
                                    .fragModule = atlasShaderModule,
                                    .blendState = &lyra::blend::Additive,
                                    .targetFormat = atlasFormat,
                                    .bindGroupLayouts = {drawBindGroupLayout}},
                                    "AtlasPipeline");

    
    wgpu::SamplerDescriptor samplerDescriptor;
    samplerDescriptor.minFilter = wgpu::FilterMode::Nearest;
    samplerDescriptor.magFilter = wgpu::FilterMode::Nearest;
    atlasSampler = device.CreateSampler(&samplerDescriptor);


    std::string drawShader = ReadTextFile("vector/shaders/draw.wgsl");
    wgpu::ShaderModule drawShaderModule = utils::CreateShaderModule(device, drawShader.c_str(), "DrawShader");
    drawPipeline = CreateRenderPipeline(device,
                                    {.vertModule = drawShaderModule,
                                    .fragModule = drawShaderModule,
                                    .blendState = &lyra::blend::OneMinusSrcAlpha,
                                    .targetFormat = wgpu::TextureFormat::RGBA8Unorm,
                                    .bindGroupLayouts = {drawBindGroupLayout, atlasBindGroupLayout}},
                                    "DrawPipeline");


    queryContainer.Init(device, 2);


    Uniforms uniformData;
    uniformData.tileSize = TILE_SIZE;
    uniformData.viewSize = uint2(IMAGE_WIDTH, IMAGE_HEIGHT);
    uniformBuffer = utils::CreateBufferFromData(device, &uniformData, sizeof(Uniforms), wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst, "UniformData");
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
    descriptor.size.width = IMAGE_WIDTH;
    descriptor.size.height = IMAGE_HEIGHT;
    // descriptor.size.depthOrArrayLayers = ;
    descriptor.dimension = wgpu::TextureDimension::e2D;
    descriptor.sampleCount = 1;
    descriptor.format = format;
    descriptor.mipLevelCount = 1;
    descriptor.usage = wgpu::TextureUsage::CopySrc | wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::RenderAttachment;
    return device.CreateTexture(&descriptor);
}

void Renderer::Render(uint32_t atlasIndices, uint32_t drawSpans) {
    if (atlasIndices + drawSpans == 0u) {
        return;
    }

    queryContainer.Reset();
    wgpu::CommandEncoder encoder = device.CreateCommandEncoder();

    if (atlasIndices > 0u) {
        utils::ComboRenderPassDescriptor atlasPassDescriptor({atlasTextureView});
        atlasPassDescriptor.cColorAttachments[0].loadOp = wgpu::LoadOp::Clear;
        atlasPassDescriptor.cColorAttachments[0].storeOp = wgpu::StoreOp::Store;
        wgpu::RenderPassEncoder atlasPass = encoder.BeginRenderPass(&atlasPassDescriptor);
        atlasPass.SetBindGroup(0, bindGroup);
        atlasPass.SetPipeline(atlasPipeline);
        atlasPass.Draw(atlasIndices * 6u);
        atlasPass.End();
    }

    if (drawSpans > 0u) {
        // wgpu::RenderPassTimestampWrites writes;
        // writes.beginningOfPassWriteIndex = 0;
        // writes.endOfPassWriteIndex = 1;
        // writes.querySet = queryContainer.querySet;

        utils::ComboRenderPassDescriptor drawDescriptor({drawTextureView});
        // drawDescriptor.timestampWrites = &writes;
        drawDescriptor.cColorAttachments[0].loadOp = wgpu::LoadOp::Clear;
        wgpu::RenderPassEncoder drawPass = encoder.BeginRenderPass(&drawDescriptor);
        drawPass.SetBindGroup(0, bindGroup);
        drawPass.SetBindGroup(1, atlasBindGroup);
        drawPass.SetPipeline(drawPipeline);
        drawPass.Draw(drawSpans * 6u);
        drawPass.End();
        // queryContainer.Resolve(encoder);
    }

    wgpu::CommandBuffer commandBuffer = encoder.Finish();
    device.GetQueue().Submit(1, &commandBuffer);

    // utils::BusyWaitDevice(device);
    // queryContainer.Read(device);

    // std::cout << queryContainer.GetTotal() << std::endl;
    // utils::BusyWaitDevice(device);
    // WriteAtlasTexture();
    // WriteColorTexture();
}

void Renderer::WriteColorTexture() const {
    uint32_t width = IMAGE_WIDTH;
    uint32_t height = IMAGE_HEIGHT;
    wgpu::Buffer buf = ReadBackTexture(device, drawTexture, width, height, 4);
    const uint8_t* data = static_cast<const uint8_t*>(buf.GetConstMappedRange());
    
    uint32_t bpr = (((width * 4) + 255) / 256) * 256;
    stbi_write_png("color.png", width, height, 4, static_cast<const void*>(data), bpr);
    buf.Destroy();
}

void Renderer::WriteAtlasTexture() const {
    uint32_t width = IMAGE_WIDTH;
    uint32_t height = IMAGE_HEIGHT;
    wgpu::Buffer buf = ReadBackTexture(device, atlasTexture, width, height, 4);
    const float* data = static_cast<const float*>(buf.GetConstMappedRange());

    for (int i = 0; i < width * height; i++) {
        float area = data[i];
        float a = std::min(std::abs(area - 2.0f * std::round(0.5f * area)), 1.0f);
        outAtlas[i] = static_cast<uint8_t>(a * 255.0f); 
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
        {4, atlasIndicesBuffer},
        {5, uniformBuffer}
    });

    atlasBindGroup = utils::MakeBindGroup(device, atlasBindGroupLayout, {
        {0, atlasTextureView},
        {1, atlasSampler},
    });
}


void Renderer::Upload(
    const std::vector<uint32_t>& colors,
    const std::vector<VPoint>& flatPoints,
    const std::vector<uint32_t>& indices,
    const std::vector<DrawSpan>& drawSpans,
    const std::vector<uint32_t>& atlasIndices,
    uint32_t numFlatPoints,
    uint32_t numIndices,
    uint32_t numDrawSpans) {
    uploadAmount = 0u;    
    CreateOrUploadBuffer(device, &pathInfoBuffer, colors, colors.size(), "ColorBuffer");
    CreateOrUploadBuffer(device, &lineIndexBuffer, indices, numIndices, "LineIndexBuffer");
    CreateOrUploadBuffer(device, &flatLinePointBuffer, flatPoints, numFlatPoints, "FlatPointBuffer");
    CreateOrUploadBuffer(device, &drawSpansBuffer, drawSpans, numDrawSpans, "DrawSpansBuffer");
    CreateOrUploadBuffer(device, &atlasIndicesBuffer, atlasIndices, atlasIndices.size(), "AtlasIndicesBuffer");

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
