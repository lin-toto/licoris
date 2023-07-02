cmake_minimum_required(VERSION 3.10)
project(embedLIC)
include(ExternalProject)

set(CMAKE_CXX_STANDARD 17)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-g -Wno-subobject-linkage -Wno-attributes")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wno-subobject-linkage -Wno-attributes")

add_custom_target(vendor)
ExternalProject_Add(
        argparse
        URL https://github.com/p-ranav/argparse/archive/refs/tags/v2.2.zip
        CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/vendor/argparse
)
add_dependencies(vendor argparse)

find_package(CUDA)
find_package(Threads REQUIRED)
find_package(OpenCV)
find_library(TensorRT_LIB nvinfer)
if (WIN32)
    find_library(SetupAPI_LIB setupapi)
endif()

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    link_libraries(OpenMP::OpenMP_CXX)
endif()

include_directories(
        ${CMAKE_SOURCE_DIR}
        $ENV{INCLUDE_DIR}
        ${CMAKE_BINARY_DIR}/vendor/argparse/include
)
link_directories(
        $ENV{LINK_DIR}
        ${CMAKE_BINARY_DIR}/vendor/argparse/lib
)

if(OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})
    link_libraries(${OpenCV_LIBS})
endif()

add_custom_target(main)

if(CUDA_FOUND AND TensorRT_LIB)
    include_directories(${CUDA_INCLUDE_DIRS})
    link_libraries(${CMAKE_THREAD_LIBS_INIT} ${CUDA_LIBRARIES} nvinfer)

    add_executable(bmshj2018_factorized_compressor bmshj2018_factorized_compressor.cpp entropy_coder/entropy_coder.cpp
            libs/tensorrt_wrapper.cpp libs/tensorrt_logger.cpp libs/utils.cpp libs/network.cpp libs/img_utils.cpp
            libs/memory.cpp libs/prealloc_pool.cpp
            pipelines/bmshj2018_factorized_pipelines.cpp entropy_coder/rans_interface.cpp)
    add_dependencies(bmshj2018_factorized_compressor vendor)
    add_dependencies(main bmshj2018_factorized_compressor)

    add_executable(bmshj2018_hyperprior_compressor bmshj2018_hyperprior_compressor.cpp entropy_coder/entropy_coder.cpp
            libs/tensorrt_wrapper.cpp libs/tensorrt_logger.cpp libs/utils.cpp libs/network.cpp libs/img_utils.cpp
            libs/memory.cpp libs/prealloc_pool.cpp
            pipelines/bmshj2018_hyperprior_pipelines.cpp entropy_coder/rans_interface.cpp)
    add_dependencies(bmshj2018_hyperprior_compressor vendor)
    add_dependencies(main bmshj2018_hyperprior_compressor)

    add_executable(bmshj2018_factorized_decompressor bmshj2018_factorized_decompressor.cpp entropy_coder/entropy_coder.cpp
            libs/tensorrt_wrapper.cpp libs/tensorrt_logger.cpp libs/utils.cpp libs/network.cpp libs/img_utils.cpp
            libs/memory.cpp libs/prealloc_pool.cpp
            pipelines/bmshj2018_factorized_pipelines.cpp entropy_coder/rans_interface.cpp)
    add_dependencies(bmshj2018_factorized_decompressor vendor)
    add_dependencies(main bmshj2018_factorized_decompressor)

    add_executable(bmshj2018_hyperprior_decompressor bmshj2018_hyperprior_decompressor.cpp entropy_coder/entropy_coder.cpp
            libs/tensorrt_wrapper.cpp libs/tensorrt_logger.cpp libs/utils.cpp libs/network.cpp libs/img_utils.cpp
            libs/memory.cpp libs/prealloc_pool.cpp
            pipelines/bmshj2018_hyperprior_pipelines.cpp entropy_coder/rans_interface.cpp)
    add_dependencies(bmshj2018_hyperprior_decompressor vendor)
    add_dependencies(main bmshj2018_hyperprior_decompressor)

    add_executable(cam_encoder cam_encoder.cpp entropy_coder/entropy_coder.cpp
            libs/tensorrt_wrapper.cpp libs/tensorrt_logger.cpp libs/utils.cpp libs/network.cpp libs/img_utils.cpp
            libs/memory.cpp libs/prealloc_pool.cpp
            pipelines/bmshj2018_factorized_pipelines.cpp entropy_coder/rans_interface.cpp)
    add_dependencies(cam_encoder vendor)
    add_dependencies(main cam_encoder)

    add_executable(cam_decoder cam_decoder.cpp entropy_coder/entropy_coder.cpp
            libs/tensorrt_wrapper.cpp libs/tensorrt_logger.cpp libs/utils.cpp libs/network.cpp libs/img_utils.cpp
            libs/memory.cpp libs/prealloc_pool.cpp
            pipelines/bmshj2018_factorized_pipelines.cpp entropy_coder/rans_interface.cpp)
    add_dependencies(cam_decoder vendor)
    add_dependencies(main cam_decoder)
else()
    message(WARNING "Unable to compile main module since one ore more dependencies are not satisfied!")
    add_compile_definitions(NO_GPU_LIBRARIES=1)
endif()