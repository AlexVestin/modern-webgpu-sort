
FILES="ComputeUtil.cpp vector/Flatten.cpp vector/path/LTRBRect.cpp vector/path/SVGUtil.cpp vector/path/VPaint.cpp vector/path/VPath.cpp vector/path/VPoint.cpp vector/vector.cpp"
FLAGS="-O3 -g -s INITIAL_MEMORY=128MB -s LLD_REPORT_UNDEFINED --preload-file paper-1.svg -s MODULARIZE=1 -s EXPORT_ES6=1 -s EXPORT_NAME=Manager -s ENVIRONMENT=web --bind --no-entry"
KURBO="./kurbo_bridge/target/wasm32-unknown-emscripten/release/libkurbo_bridge.a"
em++ $KURBO $FILES -o main.html $FLAGS -I./dependencies/dawn/src
