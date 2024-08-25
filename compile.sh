

PRELOAD="--preload-file paper-1.svg --preload-file vector/shaders/atlas.wgsl --preload-file vector/shaders/draw.wgsl --preload-file vector/shaders/points.wgsl"
WEBGPU_FILES="vector/Renderer.cpp wgpu/ComboRenderPipelineDescriptor.cpp wgpu/WGPUHelpers.cpp"
FILES="vector/bindings.cpp ComputeUtil.cpp vector/Flatten.cpp vector/path/LTRBRect.cpp vector/path/SVGUtil.cpp vector/path/VPaint.cpp vector/path/VPath.cpp vector/path/VPoint.cpp vector/vector.cpp"
FLAGS="-O3 -s USE_WEBGPU=1 -s INITIAL_MEMORY=128MB -s LLD_REPORT_UNDEFINED -s MODULARIZE=1 -s EXPORT_ES6=1 -s EXPORT_NAME=Manager -s ENVIRONMENT=web --bind --no-entry"
KURBO="./kurbo_bridge/target/wasm32-unknown-emscripten/release/libkurbo_bridge.a"
em++ $KURBO $WEBGPU_FILES $FILES -o main.js $PRELOAD $FLAGS -I./dependencies/dawn/src
