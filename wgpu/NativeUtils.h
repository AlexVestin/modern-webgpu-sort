#pragma once

#include <dawn/dawn_proc.h>
#include <dawn/native/DawnNative.h>
#include <memory>

#include <webgpu/webgpu_cpp.h>

#ifndef _WIN32
#include <unistd.h>
#endif


void DumpAllInfo();

namespace NativeUtils {
wgpu::Device SetupDevice(const std::unique_ptr<wgpu::Instance>& instance, const wgpu::Adapter& adapter);
wgpu::Adapter SetupAdapter(const std::unique_ptr<wgpu::Instance>& instance);
}  // namespace NativeUtils