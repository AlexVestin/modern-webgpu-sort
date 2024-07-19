#include "NativeUtils.h"

#include <iostream>

static WGPUDevice backendDevice;
static DawnProcTable backendProcs;

namespace NativeUtils {
#if __APPLE__
const wgpu::BackendType backendType = wgpu::BackendType::Metal;
wgpu::TextureFormat textureFormat = wgpu::TextureFormat::BGRA8Unorm;
#else
const wgpu::BackendType backendType = wgpu::BackendType::Vulkan;
#if ANDROID
wgpu::TextureFormat textureFormat = wgpu::TextureFormat::RGBA8Unorm;
#else
wgpu::TextureFormat textureFormat = wgpu::TextureFormat::BGRA8Unorm;
#endif
#endif

static void PrintDeviceLoss(const WGPUDevice* device, WGPUDeviceLostReason reason, const char* message,
                            void* userdata) {
    const char* reasonName = "";
    switch (reason) {
        case WGPUDeviceLostReason_Unknown:
            reasonName = "Unknown";
            break;
        case WGPUDeviceLostReason_Destroyed:
            reasonName = "Destroyed";
            break;
        case WGPUDeviceLostReason_InstanceDropped:
            reasonName = "InstanceDropped";
            break;
        case WGPUDeviceLostReason_FailedCreation:
            reasonName = "FailedCreation";
            break;
        default:
            std::cerr << "-----UNREACHABLE-----" << std::endl;
    }
    std::cerr << "Device lost because of " << reasonName << ": " << message << std::endl;
}

static void PrintDeviceError(WGPUErrorType errorType, const char* message, void*) {
    const char* errorTypeName = "";
    switch (errorType) {
        case WGPUErrorType_Validation:
            errorTypeName = "Validation";
            break;
        case WGPUErrorType_OutOfMemory:
            errorTypeName = "Out of memory";
            break;
        case WGPUErrorType_Unknown:
            errorTypeName = "Unknown";
            break;
        case WGPUErrorType_DeviceLost:
            errorTypeName = "Device lost";
            break;
        default:
            std::cerr << "Unknown error in PrintDeviceError" << std::endl;
            break;
    }
    std::cerr << errorTypeName << " error: " << message << std::endl;
}

wgpu::Adapter SetupAdapter(const std::unique_ptr<wgpu::Instance>& instance) {
    wgpu::RequestAdapterOptions options = {};
    options.backendType = backendType;
    options.powerPreference = wgpu::PowerPreference::HighPerformance;

    wgpu::Adapter adapter;
    instance->WaitAny(
        instance->RequestAdapter(
            &options, {nullptr, wgpu::CallbackMode::WaitAnyOnly,
                       [](WGPURequestAdapterStatus status, WGPUAdapter adapter, const char* message, void* userdata) {
                           if (status != WGPURequestAdapterStatus_Success) {
                               std::cerr << "Failed to get an adapter:" << message << std::endl;
                               return;
                           }
                           *static_cast<wgpu::Adapter*>(userdata) = wgpu::Adapter::Acquire(adapter);
                       },
                       &adapter}),
        UINT64_MAX);

    return adapter;
}

wgpu::Device SetupDevice(const std::unique_ptr<wgpu::Instance>& instance, const wgpu::Adapter& adapter) {
    if (adapter == nullptr) {
        std::cerr << "Failed to create adapter" << std::endl;
        exit(1);
    }

    wgpu::AdapterProperties properties;
    adapter.GetProperties(&properties);
    std::cout << "Using adapter \"" << properties.name << "\"" << std::endl;

    // Synchronously request the device.
    wgpu::DeviceDescriptor deviceDesc;
    deviceDesc.uncapturedErrorCallbackInfo = {nullptr, PrintDeviceError, nullptr};
    deviceDesc.deviceLostCallbackInfo = {nullptr, wgpu::CallbackMode::AllowSpontaneous, PrintDeviceLoss, nullptr};

    std::vector<wgpu::FeatureName> requiredFeatures = {wgpu::FeatureName::TimestampQuery, wgpu::FeatureName::Subgroups};
    deviceDesc.requiredFeatures = requiredFeatures.data();
    deviceDesc.requiredFeatureCount = requiredFeatures.size();

    wgpu::RequiredLimits limits;
    limits.nextInChain = nullptr;
    limits.limits.maxStorageBuffersPerShaderStage = 10;
    limits.limits.maxBufferSize = 1u << 30u;
    limits.limits.maxStorageBufferBindingSize = 1u << 30u;
    deviceDesc.requiredLimits = &limits;

    std::cout << "MaxBufferSize: " << limits.limits.maxBufferSize << std::endl; 

    wgpu::Device device;
    instance->WaitAny(
        adapter.RequestDevice(
            &deviceDesc, {nullptr, wgpu::CallbackMode::WaitAnyOnly,
                          [](WGPURequestDeviceStatus status, WGPUDevice device, const char* message, void* userdata) {
                              if (status != WGPURequestDeviceStatus_Success) {
                                  std::cerr << "Failed to get an device:" << message << std::endl;
                                  exit(1);
                              }
                              *static_cast<wgpu::Device*>(userdata) = wgpu::Device::Acquire(device);
                          },
                          &device}),
        UINT64_MAX);

    wgpu::SupportedLimits supportedLimits;
    device.GetLimits(&supportedLimits);

    std::cout << "Supported: " << supportedLimits.limits.maxBufferSize << " " << supportedLimits.limits.maxStorageBufferBindingSize << std::endl;

    return device;
}
}  // namespace NativeUtils