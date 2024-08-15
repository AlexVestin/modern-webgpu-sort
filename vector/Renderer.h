#pragma once

#include <iostream>

#include "../wgpu/NativeUtils.h"


static std::unique_ptr<wgpu::Instance> instance;

static const wgpu::BufferUsage storageUsage = wgpu::BufferUsage::Storage;
static const wgpu::BufferUsage copyDstUsage = storageUsage | wgpu::BufferUsage::CopyDst;
static const wgpu::BufferUsage copySrcUsage = storageUsage | wgpu::BufferUsage::CopySrc;
static const wgpu::BufferUsage copyAllUsage = copySrcUsage | copyDstUsage;

class Renderer {
public:
    void Init() {

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

        // wgpu::BindGroupLayout drawLayout = utils::MakeBindGroupLayout(device, "DrawBindGroupLayout", {
        //     {0, wgpu::ShaderStage::Vertex | wgpu::ShaderStage::Fragment, wgpu::BufferBindingType::ReadOnlyStorage},
        //     {1, wgpu::ShaderStage::Vertex | wgpu::ShaderStage::Fragment, wgpu::BufferBindingType::ReadOnlyStorage},
        //     {2, wgpu::ShaderStage::Vertex | wgpu::ShaderStage::Fragment, wgpu::BufferBindingType::ReadOnlyStorage},
        //     {3, wgpu::ShaderStage::Vertex | wgpu::ShaderStage::Fragment, wgpu::BufferBindingType::ReadOnlyStorage},
        // });

        // wgpu::BindGroup bindGroup = utils::MakeBindGroup(device, drawLayout, {
        //     {0, pathInfoBuffer},
        //     {1, lineIndexBuffer},
        //     {2, flatLinePointBuffer},
        //     {3, drawSpansBuffer},
        // });

        // wgpu::PipelineLayoutDescriptor descriptor;
    }


    void Upload() {
        // Save image to disk
        // pathInfoBuffer =
        //     utils::CreateBufferFromData(device, colors.data(), colors.size() * sizeof(uint32_t), copyDstUsage, "PathInformation");
        
        // lineIndexBuffer =
        //     utils::CreateBufferFromData(device, indices.data(), indices.size() * sizeof(uint32_t), copyDstUsage, "LineIndices");
        
        // flatLinePointBuffer = 
        //     utils::CreateBufferFromData(device, flatLinePoints.data(), flatLinePoints.size() * sizeof(VPoint), copyDstUsage, "flatLinePoints");;

        // drawSpansBuffer = 
        //     utils::CreateBufferFromData(device, drawSpans.data(), drawSpans.size() * sizeof(DrawSpan), copyDstUsage, "DrawSpans");;


    }

    void Dispose() {
        device.Destroy();
    }

private:
    wgpu::Buffer pathInfoBuffer;
    wgpu::Buffer lineIndexBuffer;
    wgpu::Buffer flatLinePointBuffer;
    wgpu::Buffer drawSpansBuffer;
    wgpu::BindGroup drawBindGroup;
    wgpu::Device device;
};