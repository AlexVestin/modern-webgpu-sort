#pragma once

#include <iostream>
#include <string>

#include "defs.h"

#include "path/VPoint.h"
#include "../wgpu/NativeUtils.h"
#include "../ComputeUtil.h"
#include "../wgpu/ComboRenderPipelineDescriptor.h"
#include "../wgpu/Blends.h"

static std::unique_ptr<wgpu::Instance> instance;

static const wgpu::BufferUsage storageUsage = wgpu::BufferUsage::Storage;
static const wgpu::BufferUsage copyDstUsage = storageUsage | wgpu::BufferUsage::CopyDst;
static const wgpu::BufferUsage copySrcUsage = storageUsage | wgpu::BufferUsage::CopySrc;
static const wgpu::BufferUsage copyAllUsage = copySrcUsage | copyDstUsage;

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
    const std::vector<const wgpu::BindGroupLayout> bindGroupLayouts = {};
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

class Renderer {
public:
    Renderer(uint32_t atlasWidth, uint32_t atlasHeight): atlasWidth{atlasWidth}, atlasHeight{atlasHeight} {
        InitDevice();
        
        drawBindGroupLayout = utils::MakeBindGroupLayout(device, "DrawBindGroupLayout", {
            {0, wgpu::ShaderStage::Vertex | wgpu::ShaderStage::Fragment, wgpu::BufferBindingType::ReadOnlyStorage},
            {1, wgpu::ShaderStage::Vertex | wgpu::ShaderStage::Fragment, wgpu::BufferBindingType::ReadOnlyStorage},
            {2, wgpu::ShaderStage::Vertex | wgpu::ShaderStage::Fragment, wgpu::BufferBindingType::ReadOnlyStorage},
            {3, wgpu::ShaderStage::Vertex | wgpu::ShaderStage::Fragment, wgpu::BufferBindingType::ReadOnlyStorage},
            {4, wgpu::ShaderStage::Vertex | wgpu::ShaderStage::Fragment, wgpu::BufferBindingType::ReadOnlyStorage},
        });

        // Create empty buffers
        pathInfoBuffer  = utils::CreateBuffer(device, 4, copyDstUsage, "ColorBuffer");
        lineIndexBuffer  = utils::CreateBuffer(device, 4, copyDstUsage, "LineIndexBuffer");
        flatLinePointBuffer  = utils::CreateBuffer(device, 4, copyDstUsage, "FlatLinePointBuffer");
        drawSpansBuffer  = utils::CreateBuffer(device, 4, copyDstUsage, "DrawSpansBuffer");
        atlasIndicesBuffer  = utils::CreateBuffer(device, 4, copyDstUsage, "AtlasIndicesBuffer");

        std::string atlasShader = ReadTextFile("vector/shaders/atlas.wgsl");
        wgpu::ShaderModule shaderModule = utils::CreateShaderModule(device, atlasShader.c_str(), "AtlasShader");
        pipeline = CreateRenderPipeline(device,
                                        {.vertModule = shaderModule,
                                        .fragModule = shaderModule,
                                        .targetFormat = wgpu::TextureFormat::R16Float,
                                        .blendState = &lyra::blend::Additive,
                                        .bindGroupLayouts = {drawBindGroupLayout}},
                                        "AtlasPipeline");
        

        wgpu::RenderPipelineDescriptor descriptor;
    }

    void InitDevice() {
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

    void CreateBindGroup() {
        bindGroup = utils::MakeBindGroup(device, drawBindGroupLayout, {
            {0, pathInfoBuffer},
            {1, lineIndexBuffer},
            {2, flatLinePointBuffer},
            {3, drawSpansBuffer},
            {4, atlasIndicesBuffer}
        });
    }

    template <typename T>
    void CreateOrUploadBuffer(const wgpu::Device& device, wgpu::Buffer* buffer, const std::vector<T>& data, const char* label) {
        uint32_t dataByteSize = data.size() * sizeof(T);
        if (dataByteSize > buffer->GetSize()) {
            // buffer->Destroy();
            *buffer = utils::CreateBufferFromData(device, data.data(), dataByteSize, copyDstUsage, label);
            needsRecreateBindGroup = true;
        } else {
            device.GetQueue().WriteBuffer(*buffer, 0, data.data(), dataByteSize);
        }

        uploadAmount += dataByteSize;
    }

    void Upload(
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

    void Dispose() {
        pathInfoBuffer.Destroy();
        lineIndexBuffer.Destroy();
        flatLinePointBuffer.Destroy();
        drawSpansBuffer.Destroy();
        atlasIndicesBuffer.Destroy();
        device.Destroy();
    }

private:
    wgpu::Buffer pathInfoBuffer;
    wgpu::Buffer lineIndexBuffer;
    wgpu::Buffer flatLinePointBuffer;
    wgpu::Buffer drawSpansBuffer;
    wgpu::Buffer atlasIndicesBuffer;
    wgpu::Device device;


    bool needsRecreateBindGroup = true;

    wgpu::RenderPipeline pipeline;

    wgpu::Texture atlasTexture;
    wgpu::TextureView atlasTextureView;

    wgpu::BindGroup bindGroup;
    wgpu::BindGroupLayout drawBindGroupLayout;

    uint32_t atlasWidth;
    uint32_t atlasHeight;

    uint32_t uploadAmount = 0u;
};