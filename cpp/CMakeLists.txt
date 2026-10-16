cmake_minimum_required(VERSION 3.18)
project(cuda_examples LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 86)

# 包含公共头文件目录
include_directories(${CMAKE_SOURCE_DIR}/common)

# ---------------------------------------------------------------------------
# 编号示例（examples/）
#
# examples/ 下按**编程模型**分两个子目录：
#   examples/simt/  —— 线程级编程（threadIdx / blockIdx），CUDA 编程的基础功
#   examples/tile/  —— 块级编程；C++ 目前没有用户级入口，见该目录 README
#
# 目标名 = <模型>_<文件名>：
#   cpp/examples/simt/00-add.cu  →  ./build/simt_00-add
#
# 加模型前缀有两个原因：两个子目录里可能出现同名文件；编号在各自模型内独立计数。
# ---------------------------------------------------------------------------
foreach(MODEL simt tile)
    file(GLOB MODEL_SOURCES "examples/${MODEL}/*.cu")

    foreach(SOURCE_FILE ${MODEL_SOURCES})
        # 获取文件名（不含扩展名）作为可执行文件名称
        get_filename_component(EXE_NAME ${SOURCE_FILE} NAME_WE)

        # 添加可执行目标
        add_executable(${MODEL}_${EXE_NAME} ${SOURCE_FILE})

        # 链接 CUDA 库
        target_link_libraries(${MODEL}_${EXE_NAME} ${CUDA_LIBRARIES})
    endforeach()
endforeach()

# ---------------------------------------------------------------------------
# 历史学习文件（legacy/）
#
# 早期随手写的 9 个示例，保留原样以便回查。目标名统一加 `legacy_` 前缀，
# 避免和 examples/ 里的编号示例重名。不需要时用 -DBUILD_LEGACY=OFF 跳过。
# ---------------------------------------------------------------------------
option(BUILD_LEGACY "Build the legacy examples in cpp/legacy" ON)

if(BUILD_LEGACY)
    file(GLOB LEGACY_SOURCES "legacy/*.cu")
    foreach(SOURCE_FILE ${LEGACY_SOURCES})
        get_filename_component(EXE_NAME ${SOURCE_FILE} NAME_WE)
        add_executable(legacy_${EXE_NAME} ${SOURCE_FILE})
        target_link_libraries(legacy_${EXE_NAME} ${CUDA_LIBRARIES})
    endforeach()
endif()