#pragma once

#include <iostream>

#include "defs.h"

#include "path/VPoint.h"
#include "../wgpu/NativeUtils.h"
#include "../ComputeUtil.h"

static std::unique_ptr<wgpu::Instance> instance;

static const wgpu::BufferUsage storageUsage = wgpu::BufferUsage::Storage;
static const wgpu::BufferUsage copyDstUsage = storageUsage | wgpu::BufferUsage::CopyDst;
static const wgpu::BufferUsage copySrcUsage = storageUsage | wgpu::BufferUsage::CopySrc;
static const wgpu::BufferUsage copyAllUsage = copySrcUsage | copyDstUsage;

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
        // wgpu::PipelineLayoutDescriptor descriptor;
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

        // Create empty buffers
        pathInfoBuffer  = utils::CreateBuffer(device, 4, copyDstUsage, "ColorBuffer");
        lineIndexBuffer  = utils::CreateBuffer(device, 4, copyDstUsage, "LineIndexBuffer");
        flatLinePointBuffer  = utils::CreateBuffer(device, 4, copyDstUsage, "FlatLinePointBuffer");
        drawSpansBuffer  = utils::CreateBuffer(device, 4, copyDstUsage, "DrawSpansBuffer");
        atlasIndicesBuffer  = utils::CreateBuffer(device, 4, copyDstUsage, "AtlasIndicesBuffer");
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