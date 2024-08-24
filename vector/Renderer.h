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

struct QueryContainer {
    void Init(const wgpu::Device& device, uint32_t numQueries) {
        this->numQueries = numQueries;

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
            uint64_t t0 = queryData[i * 2];
            uint64_t t1 = queryData[i * 2 + 1];
            gpu_times[i] += t1 - t0;
        }
    }

    const std::vector<uint64_t>& GetTimings() const { return gpu_times; }

    uint64_t GetTotal() const {
        uint64_t total = 0u;
        for (int i = 0; i < numQueries; i++) {
            total += gpu_times[i];
        }
        return total;
    }

    double GetTotalMs() const {
        uint64_t total = 0u;
        for (int i = 0; i < numQueries; i++) {
            total += gpu_times[i];
        }
        return total / 1000000.0;
    }


    void Reset() {
        for (int i = 0; i < gpu_times.size(); i++) {
            gpu_times[i] = 0u;
        }
    }

    std::vector<uint64_t> gpu_times;
    uint32_t numQueries;
    wgpu::Buffer buffer;
    wgpu::QuerySet querySet;
};

class Renderer {
public:
    Renderer(uint32_t atlasWidth, uint32_t atlasHeight);
    
    void InitDevice();


    wgpu::Texture CreateTexture(const wgpu::TextureFormat format) const;

    void Render(uint32_t atlasIndices, uint32_t drawSpans, uint32_t numPoints);

    void WriteAtlasTexture() const;
    void WriteColorTexture() const;

    void CreateBindGroup();

    template <typename T>
    void CreateOrUploadBuffer(const wgpu::Device& device, wgpu::Buffer* buffer, const std::vector<T>& data, uint32_t size, const char* label) {
        uint32_t dataByteSize = size * sizeof(T);
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
        const std::vector<uint32_t>& atlasIndices,
        uint32_t numFlatPoints, 
        uint32_t numIndices,
        uint32_t numDrawSpans);

    void Dispose();

private:
    wgpu::Buffer pathInfoBuffer;
    wgpu::Buffer lineIndexBuffer;
    wgpu::Buffer flatLinePointBuffer;
    wgpu::Buffer drawSpansBuffer;
    wgpu::Buffer atlasIndicesBuffer;
    wgpu::Buffer uniformBuffer;

    wgpu::Buffer textureDataBuffer;
    wgpu::Device device;


    bool needsRecreateBindGroup = true;

    wgpu::RenderPipeline atlasPipeline;
    wgpu::RenderPipeline drawPipeline;
    wgpu::RenderPipeline pointsPipeline;

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

    QueryContainer queryContainer;
};