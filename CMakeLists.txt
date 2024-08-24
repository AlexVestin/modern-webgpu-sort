cmake_minimum_required(VERSION 3.3.2)

include(FetchContent)
FetchContent_Declare(
    Corrosion
    GIT_REPOSITORY https://github.com/corrosion-rs/corrosion.git
    GIT_TAG v0.4 # Optionally specify a commit hash, version tag or branch here
)
FetchContent_MakeAvailable(Corrosion)

project(main CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set (CMAKE_CXX_STANDARD 20)
# set(CMAKE_CXX_FLAGS "-g -fsanitize=address -std=c++20")

set(CMAKE_CXX_FLAGS "-std=c++20")


set(SOURCES 
  "main.cpp"
  "SegSort.cpp"
  "ComputeUtil.cpp"
  "wgpu/DawnInfo.cpp"
  "wgpu/NativeUtils.cpp"
  "wgpu/WGPUHelpers.cpp"
  "Subgroups.cpp"
)

set(VECTOR_SOURCES 
  "ComputeUtil.cpp"
  "vector/Flatten.cpp"
  "vector/path/LTRBRect.cpp"
  "vector/path/SVGUtil.cpp"
  "vector/path/VPaint.cpp"
  "vector/path/VPath.cpp"
  "vector/path/VPoint.cpp"
  "vector/vector.cpp"
)

if (NOT EMSCRIPTEN)
  list(APPEND VECTOR_SOURCES 
        "wgpu/NativeUtils.cpp"
        "wgpu/DawnInfo.cpp"
        "vector/Renderer.cpp"
    )
  add_executable(${PROJECT_NAME} ${VECTOR_SOURCES})
endif()

MESSAGE("${CMAKE_CXX_FLAGS}")



if (NOT EMSCRIPTEN)
corrosion_import_crate(MANIFEST_PATH kurbo_bridge/Cargo.toml)
set(TINT_BUILD_DOCS OFF CACHE BOOL "Enable building tint docs." FORCE)
set(TINT_BUILD_TESTS OFF CACHE BOOL "Enable building tint tests." FORCE)
set(TINT_BUILD_CMD_TOOLS OFF CACHE BOOL "Enable building tint command line tools" FORCE)
set(DAWN_BUILD_SAMPLES OFF CACHE BOOL "Enable dawn building samples" FORCE)
set(DAWN_USE_GLFW OFF CACHE BOOL "Enable building dawn glfw." FORCE)

include_directories (dependencies/dawn)
include_directories (dependencies/dawn/src)
include_directories (dependencies/dawn/include)

add_subdirectory(dependencies/dawn)
set (DAWN_LIBRARIES  
    dawn_internal_config
    dawncpp
    dawn_proc
    dawn_common
    dawn_native
)

target_link_libraries(${PROJECT_NAME} ${DAWN_LIBRARIES})
target_link_libraries(${PROJECT_NAME} kurbo_bridge)
endif()


if (EMSCRIPTEN)
    set(CMAKE_EXECUTABLE_SUFFIX ".html")
    string(CONCAT COMPILER_FLAGS ${COMPILER_FLAGS} 
        " -msimd128"
        " -msse4.1"
        " -O3"
        " -std=c++20"
    )

    message("\n-------Building web application--------\n")
    add_definitions(-DPLATFORM_WEB)

    # list(APPEND SOURCES "src/render_bindings.cpp")
    # add_executable(${CMAKE_PROJECT_NAME} ${SOURCES}) 
    add_executable(${PROJECT_NAME} ${VECTOR_SOURCES})

    include_directories (dependencies/dawn/src)
    
    string(CONCAT EMCC_LINKER_FLAGS 
            " -g"
            " -s INITIAL_MEMORY=128MB"
            " -o test.html"
            " -s LLD_REPORT_UNDEFINED"
            " --preload-file paper-1.svg"
    )

    


    set(CMAKE_EXE_LINKER_FLAGS ${EMCC_LINKER_FLAGS} CACHE INTERNAL "" FORCE)
    set(CMAKE_CXX_FLAGS ${COMPILER_FLAGS})
    MESSAGE("${CMAKE_EXE_LINKER_FLAGS}")
    
    target_link_libraries(${PROJECT_NAME} ${CMAKE_SOURCE_DIR}/kurbo_bridge/target/wasm32-unknown-emscripten/release/libkurbo_bridge.a)
endif()

set_target_properties(${CMAKE_PROJECT_NAME}
  PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)
