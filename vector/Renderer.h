#pragma once

#include <iostream>
#include <string>

#include "defs.h"

#include "path/VPoint.h"
#include "../wgpu/NativeUtils.h"
#include "../ComputeUtil.h"
#include "../wgpu/ComboRenderPipelineDescriptor.h"
#include "../wgpu/Blends.h"


static const wgpu::BufferUsage storageUsage = wgpu::BufferUsage::Storage;
static const wgpu::BufferUsage copyDstUsage = storageUsage | wgpu::BufferUsage::CopyDst;
static const wgpu::BufferUsage copySrcUsage = storageUsage | wgpu::BufferUsage::CopySrc;
static const wgpu::BufferUsage copyAllUsage = copySrcUsage | copyDstUsage;

class Renderer {
public:
    Renderer(uint32_t atlasWidth, uint32_t atlasHeight);
    
    void InitDevice();


    wgpu::Texture CreateTexture(const wgpu::TextureFormat format) const;

    void Render(uint32_t atlasIndices, uint32_t drawSpans) const;

    void WriteAtlasTexture() const;
    void WriteColorTexture() const;

    void CreateBindGroup();

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
        const std::vector<uint32_t>& atlasIndices);

    void Dispose();

private:
    wgpu::Buffer pathInfoBuffer;
    wgpu::Buffer lineIndexBuffer;
    wgpu::Buffer flatLinePointBuffer;
    wgpu::Buffer drawSpansBuffer;
    wgpu::Buffer atlasIndicesBuffer;

    wgpu::Buffer textureDataBuffer;
    wgpu::Device device;


    bool needsRecreateBindGroup = true;

    wgpu::RenderPipeline atlasPipeline;
    wgpu::RenderPipeline drawPipeline;

    wgpu::Texture atlasTexture;
    wgpu::TextureView atlasTextureView;


    wgpu::Texture drawTexture;
    wgpu::TextureView drawTextureView;

    wgpu::BindGroup bindGroup;
    wgpu::BindGroup atlasBindGroup;
    wgpu::Sampler atlasSampler;

    wgpu::BindGroupLayout drawBindGroupLayout;
    wgpu::BindGroupLayout atlasBindGroupLayout;

    uint32_t atlasWidth;
    uint32_t atlasHeight;

    uint32_t uploadAmount = 0u;
};