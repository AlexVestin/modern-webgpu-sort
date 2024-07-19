cmake_minimum_required(VERSION 3.3.2)

project(main CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set (CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "-O3 -std=c++20")

set(SOURCES 
  "main.cpp"
  "SegSort.cpp"
  "ComputeUtil.cpp"
  "wgpu/DawnInfo.cpp"
  "wgpu/NativeUtils.cpp"
  "wgpu/WGPUHelpers.cpp"
  "Subgroups.cpp"
)

MESSAGE("${CMAKE_CXX_FLAGS}")

add_executable(${PROJECT_NAME} ${SOURCES})

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

set_target_properties(${CMAKE_PROJECT_NAME}
  PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)
