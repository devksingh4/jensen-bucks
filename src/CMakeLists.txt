cmake_minimum_required(VERSION 3.18)
project(WinDirectStorage LANGUAGES CXX CUDA)

# ============================================
# C++ and CUDA Standard
# ============================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ============================================
# Optimization Flags
# ============================================
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
set(CMAKE_CUDA_ARCHITECTURES native)

# ============================================
# CUDA/NVCC Configuration
# ============================================
find_package(CUDAToolkit REQUIRED)
enable_language(CUDA)

# ============================================
# DirectStorage SDK Configuration
# ============================================
set(DSTORAGE_SDK_DIR "${CMAKE_SOURCE_DIR}/directstorage" CACHE PATH "DirectStorage SDK path")

find_path(DSTORAGE_INCLUDE_DIR
    NAMES dstorage.h
    PATHS
        "${DSTORAGE_SDK_DIR}/native/include"
        "C:/Program Files (x86)/Microsoft DirectStorage SDK/native/include"
        "$ENV{DSTORAGE_DIR}/include"
    REQUIRED
)

find_library(DSTORAGE_LIBRARY
    NAMES dstorage
    PATHS
        "${DSTORAGE_SDK_DIR}/native/lib/x64"
        "C:/Program Files (x86)/Microsoft DirectStorage SDK/native/lib/x64"
        "$ENV{DSTORAGE_DIR}/native/lib/x64"
    REQUIRED
)

add_library(DirectStorage::DirectStorage UNKNOWN IMPORTED)
set_target_properties(DirectStorage::DirectStorage PROPERTIES
    IMPORTED_LOCATION "${DSTORAGE_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${DSTORAGE_INCLUDE_DIR}"
)

# ============================================
# Your Project Targets
# ============================================
add_executable(processor
    processor.cu
)

target_link_libraries(processor PRIVATE
    CUDA::cudart
    CUDA::cuda_driver
    DirectStorage::DirectStorage
    d3d12.lib
    dxgi.lib
)

# Set CUDA-specific properties for the correct target
set_target_properties(processor PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

set(DSTORAGE_DLL_DIR "${DSTORAGE_SDK_DIR}/native/bin/x64")

add_custom_command(TARGET processor POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${DSTORAGE_DLL_DIR}/dstorage.dll"
        "$<TARGET_FILE_DIR:processor>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${DSTORAGE_DLL_DIR}/dstoragecore.dll"
        "$<TARGET_FILE_DIR:processor>"
    COMMENT "Copying DirectStorage DLLs to build directory"
)