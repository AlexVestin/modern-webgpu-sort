#include <emscripten.h>
#include <emscripten/bind.h>
#include <emscripten/html5_webgpu.h>

#include "vector.h"
using namespace emscripten;

// https://github.com/google/skia/blob/main/modules/canvaskit/WasmCommon.h#L30
using WASMPointerF32 = uintptr_t;
using WASMPointerU8  = uintptr_t;
using WASMPointerU16 = uintptr_t;
using WASMPointerU32 = uintptr_t;
using WASMPointer    = uintptr_t;
using Float32Array = emscripten::val;

EMSCRIPTEN_BINDINGS(Manager) {
    class_<VectorPathBuilder>("VectorPathBuilder")
        .constructor<>()
        .function("Zoom", &VectorPathBuilder::Zoom)
        .function("Move", &VectorPathBuilder::Move)
        .function("Load", &VectorPathBuilder::Load)
        .function("Process", &VectorPathBuilder::Process);
}
