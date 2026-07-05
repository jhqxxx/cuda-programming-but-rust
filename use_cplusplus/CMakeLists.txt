cmake_minimum_required(VERSION 3.18)
project(cuda_examples LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 86)

# 包含公共头文件目录
include_directories(${CMAKE_SOURCE_DIR}/common)

# 自动收集 examples 目录下的所有 .cu 文件
file(GLOB EXAMPLE_SOURCES "examples/*.cu")

foreach(SOURCE_FILE ${EXAMPLE_SOURCES})
    # 获取文件名（不含扩展名）作为可执行文件名称
    get_filename_component(EXE_NAME ${SOURCE_FILE} NAME_WE)
    
    # 添加可执行目标
    add_executable(${EXE_NAME} ${SOURCE_FILE})
    
    # 链接 CUDA 库
    target_link_libraries(${EXE_NAME} ${CUDA_LIBRARIES})
endforeach()