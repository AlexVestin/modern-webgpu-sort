// Copyright 2021, Alexander Vestin (alex.vestin@gmail.com)

#include <emscripten.h>
#include <emscripten/bind.h>
#include <emscripten/html5_webgpu.h>




extern "C" {
    WGPUSurface getSurface(uint32_t type);
} 

using namespace emscripten;

// https://github.com/google/skia/blob/main/modules/canvaskit/WasmCommon.h#L30
using WASMPointerF32 = uintptr_t;
using WASMPointerU8  = uintptr_t;
using WASMPointerU16 = uintptr_t;
using WASMPointerU32 = uintptr_t;
using WASMPointer    = uintptr_t;

using Float32Array = emscripten::val;

EMSCRIPTEN_BINDINGS(Manager) {
    register_vector<uint32_t>("UintList");
    register_map<uint32_t, uint32_t>("UintMap");

    class_<ItemBase>("ItemBase")
        .function("SetTime", &ItemBase::SetTime)
        .function("GetTime", &ItemBase::GetTimings)
        .function("GetId", &ItemBase::GetId);

    class_<State>("CoreState")
        .property("startTime", &State::startTime)
        .property("endTime", &State::endTime)
        .property("id", &State::id)
        .property("type", &State::itemType)
        .property("name", &State::name)
        .property("children", &State::childIds)
        .property("parent", &State::parentId)
        .property("keyframeValues", &State::keyframeValues)
        .property("keyframes", &State::keyframes)
        .property("visible", &State::visible)
        .property("hasView", &State::hasView)
        .property("isMask", &State::isMask)
        ;
    class_<DeserializeInfo>("DeserializeInfo")
        .property("idMapping", &DeserializeInfo::idMapping)
        .property("assets", &DeserializeInfo::assets);
    
    class_<float2>("float2")
        .property("x", &float2::x)
        .property("y", &float2::y);

    class_<lyra::Keyframe>("CoreKeyframe")
        .property("time", &lyra::Keyframe::time)
        .property("id", &lyra::Keyframe::id)
        .function("easing", &lyra::Keyframe::GetEasingEncoded)
        .function("GetJsValue", &lyra::Keyframe::GetJsVal)
        .function("SetEasing", &lyra::Keyframe::SetEasingTypeEncoded)
        ;

    register_vector<int>("VectorInt");
    register_vector<lyra::Keyframe>("KeyframeList");
    register_vector<State>("StateList");
    register_vector<std::vector<lyra::Keyframe>>("VectorKeyframes");
    register_vector<emscripten::val>("JSObjectVector");

    // this.projectName = p.project.n;
    // this.framerate = p.project.fr;
    // this.compositionDuration = p.project.d;
    // this.width = p.project.r.x;
    // this.height = p.project.r.y;

    class_<RenderManager>("RenderManager")
        .function("GetProjectName", optional_override([](const RenderManager& self) -> uintptr_t {
            return reinterpret_cast<uintptr_t>(self.GetProjectName());
        }), allow_raw_pointers())
        .function("GetFramerate", &RenderManager::GetFramerate)
        .function("GetCompositionDuration", &RenderManager::GetCompositionDuration)
        .function("GetViewSize", &RenderManager::GetViewSize)

        .function("SetEditMode", optional_override([](RenderManager& self, uint32_t isOn) {
            self.SetEditMode(static_cast<EditMode>(isOn));
        }))
        .function("AddSvgImage", optional_override([](RenderManager& self, uint32_t parentId, uintptr_t str, uint32_t size) {
            const char* data = reinterpret_cast<const char*>(str);
            std::string val = std::string(data, data + size);
            self.AddSvgImage(parentId, val);
        }))
        .function("GetValidParent", optional_override([](const RenderManager& self, uint32_t parentId) -> const ItemBase* {
            ItemBase* parentItem = self.FindItem(parentId);
            if (parentItem == nullptr) {
                return nullptr;
            }
           
            return self.GetValidParent(parentItem);
        }), allow_raw_pointers())
        .function("SetAudioDataPtr", optional_override([](RenderManager& self, uintptr_t dataPtr, size_t size) -> void {
            float* data = reinterpret_cast<float*>(dataPtr);
            self.SetAudioData(data, size);
        }), allow_raw_pointers())

        .function("SetShape", optional_override([](const RenderManager& self, ItemBase* item, uint32_t variation) -> void {
            if (item->GetType() == lyra::ItemType::Path) {
                Path* path = static_cast<Path*>(item);
                Shape s = static_cast<Shape>(variation);
                path->SetShape(s);
            }
        }), allow_raw_pointers())
        .function("Serialize2", optional_override([](const RenderManager& self, uintptr_t dataPtr, uintptr_t sizePtr, uintptr_t assetsString, uint32_t assetsStringSize) -> bool {
            uint8_t** data = reinterpret_cast<uint8_t**>(dataPtr);
            uint32_t* size = reinterpret_cast<uint32_t*>(sizePtr);


            const char* assetsData = reinterpret_cast<const char*>(assetsString);
            std::string val = std::string(assetsData, assetsData + assetsStringSize);
            return self.Serialize2(data, size, val);
        }), allow_raw_pointers())

        .function("Deserialize2", optional_override([](RenderManager& self, uintptr_t dataPtr, uint32_t size, bool clear) -> DeserializeInfo {
            const uint8_t* data = reinterpret_cast<const uint8_t*>(dataPtr);
            return self.Deserialize2(data, size, clear);
        }), allow_raw_pointers())

        .function("GetItem", &RenderManager::GetItem, allow_raw_pointers())
        .function("ApplyAction", &RenderManager::ApplyAction)
        .function("Dispose", &RenderManager::Dispose)
        .function("SetMediaSize", &RenderManager::SetMediaSize)
        .function("Clear", &RenderManager::Clear)
        .function("SerializeState", &RenderManager::SerializeItems)
        .function("Deserialize", &RenderManager::Deserialize)
        .function("GetKeyframe", &RenderManager::GetKeyframe)
        .function("SetKeyframeValue", &RenderManager::SetKeyframeValue)
        // .function("GetKeyframes", &RenderManager::GetItemKeyframes)
        .function("MoveKeyframes", &RenderManager::MoveKeyframes)
        .function("DeleteKeyframes", &RenderManager::DeleteKeyframes)
        .function("GetState", &RenderManager::GetState)
        .function("Update", &RenderManager::Update)
        .function("Render", optional_override([](RenderManager& self, float time, uint32_t swapchainHandle) {
            wgpu::SwapChain swapchain = wgpu::SwapChain::Acquire(emscripten_webgpu_import_swap_chain(swapchainHandle));
            self.Render(time, swapchain.GetCurrentTextureView());
        }))
        .function("CallFileCallback", optional_override([](RenderManager& self, uint32_t id, uintptr_t dataptr, size_t size) {
            const void* data = reinterpret_cast<const void*>(dataptr);
            self.CallAndEraseFileCallback(id, data, size);
        }))
        .function("beginFrame", &RenderManager::BeginFrame)
        .function("OnMouseMove", &RenderManager::OnMouseMove)
        .function("OnMouseEvent", &RenderManager::OnMouseEvent)
        .function("TrackpadMove", &RenderManager::TrackpadMove)
        .function("OnScrollEvent", &RenderManager::OnScrollEvent)
        .function("OnKeyboardEvent", &RenderManager::OnKeyboardEvent)
        .function("AddItemTest", &RenderManager::AddItemTest)
        .function("SetItemValues", optional_override([](RenderManager& self, uint32_t type, uintptr_t idsPtr, size_t idsSize, uint32_t value) {
            self.SetValue(type, idsPtr, idsSize, value);
        }))
        // .function("AddItem", &RenderManager::AddItem)
        .function("SetFramerate", &RenderManager::SetFramerate)
        .function("SetTimelineDuration", &RenderManager::SetTimelineDuration)
        .function("SetItemTimes", &RenderManager::SetItemTimes)
        .function("SetVisible", &RenderManager::SetVisible)
        .function("SetValue1f", optional_override([](RenderManager& self, uint32_t type, uintptr_t idsPtr, size_t idsSize, float value) {
            self.SetValue(type, idsPtr, idsSize, static_cast<float>(value));
        }))
        .function("SetValue2f", optional_override([](RenderManager& self, uint32_t type, uintptr_t idsPtr, size_t idsSize, float v0, float v1) {
            self.SetValue(type, idsPtr, idsSize, float2(v0, v1));
        }))
        .function("SetValue1u", optional_override([](RenderManager& self, uint32_t type, uintptr_t idsPtr, size_t idsSize, uint32_t value) {
            self.SetValue(type, idsPtr, idsSize, value);
        }))
        .function("SetValueString", optional_override([](RenderManager& self, uint32_t type, uintptr_t idsPtr, size_t idsSize, uintptr_t str, uint32_t size) {
            const char* data = reinterpret_cast<const char*>(str);
            std::string val = std::string(data, data + size);
            self.SetValue(type, idsPtr, idsSize, val);
        }))

        .function("SetValueArray", optional_override([](RenderManager& self, uint32_t type, uintptr_t idsPtr, size_t idsSize, uintptr_t str, uint32_t size) {
            const void* data = reinterpret_cast<const void*>(str);
            lyra::ArrayRep rep = {.ptr = data, .size = size};
            self.SetValue(type, idsPtr, idsSize, rep);
        }))

        .function("SetKeyframeActive", &RenderManager::SetKeyframeActive)
        .function("Paste", &RenderManager::Paste)
        .function("Group", &RenderManager::Group)
        .function("GetValue", &RenderManager::GetValue)
        .function("SetSize", &RenderManager::SetSize)
        .function("SetCanvasSize", &RenderManager::SetCanvasSize)
        .function("SetExternalTextureMapping", &RenderManager::SetExternalTextureMapping)
        .function("MoveItems", &RenderManager::MoveItems)
        .function("DeleteItems", &RenderManager::DeleteItems)
        .function("SelectItems", &RenderManager::SelectItems, allow_raw_pointers())
        .function("AddItem", &RenderManager::AddItem)
        .function("CreateItem", optional_override([](RenderManager& self, uint32_t parentId, uint32_t type) {
            ItemBase* parentItem = self.FindItem(parentId);
            ItemBase* newItem = self.CreateItem(parentItem, type);
            return newItem;
        }), allow_raw_pointers())

        .function("GetSelectedItemIds", &RenderManager::GetSelectedItemIds)
        .function("InsertItem", optional_override([](RenderManager& self, uint32_t parentId, ItemBase* item) {
            ItemBase* parentItem = self.FindItem(parentId);
            parentItem->AddChild(item);
            self.SetSelectedItems({item});
            self.CreateState();
        }), allow_raw_pointers());

    function("CreateSwapchain", optional_override([](uint32_t canvasWidth, uint32_t canvasHeight, uint32_t type) {
        WGPUDevice device = emscripten_webgpu_get_device();
        WGPUSurface surface = getSurface(type); 
        WGPUSwapChainDescriptor scDesc;
        scDesc.nextInChain = nullptr;
        scDesc.usage = WGPUTextureUsage_RenderAttachment;
        scDesc.format = WGPUTextureFormat_BGRA8Unorm;
        scDesc.width = canvasWidth;
        scDesc.height = canvasHeight;
        scDesc.presentMode = WGPUPresentMode_Fifo; 
        WGPUSwapChain swapchain = wgpuDeviceCreateSwapChain(device, surface, &scDesc);
        return emscripten_webgpu_export_swap_chain(swapchain);
    }));

    function("CreateRenderManager", optional_override([](uint32_t width, uint32_t height, uint32_t canvasWidth, uint32_t canvasHeight, uint32_t featureFlags) -> RenderManager* {
        WGPUDevice device = emscripten_webgpu_get_device();

        return new RenderManager(
            wgpu::Device::Acquire(device), 
            wgpu::TextureFormat::BGRA8Unorm, 
            uint2{width, height},
            uint2{canvasWidth, canvasHeight},
            featureFlags
        );
    }),  allow_raw_pointers());
}
