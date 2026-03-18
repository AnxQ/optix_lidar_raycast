cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(optix_lidar_raycast LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)

# Find OptiX
set(OptiX_INSTALL_DIR "/mnt/private/NVIDIA-OptiX-SDK-8.1.0-linux64-x86_64" CACHE PATH "Path to OptiX SDK")
find_path(OptiX_INCLUDE_DIR
    NAMES optix.h
    PATHS ${OptiX_INSTALL_DIR}/include
)

if(NOT OptiX_INCLUDE_DIR)
    message(FATAL_ERROR "OptiX SDK not found. Please set OptiX_INSTALL_DIR")
endif()

message(STATUS "OptiX found at: ${OptiX_INCLUDE_DIR}")

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OptiX_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Create a combined CUDA file for PTX compilation
set(CUDA_KERNEL_COMBINED ${CMAKE_CURRENT_BINARY_DIR}/optix_kernels_combined.cu)

# Combine all kernel files
file(WRITE ${CUDA_KERNEL_COMBINED}
    "#include \"${CMAKE_CURRENT_SOURCE_DIR}/include/LaunchParams.h\"\n"
)

foreach(kernel_file raygen.cu closesthit.cu miss.cu)
    file(READ ${CMAKE_CURRENT_SOURCE_DIR}/src/${kernel_file} kernel_content)
    file(APPEND ${CUDA_KERNEL_COMBINED} "\n// From ${kernel_file}\n${kernel_content}\n")
endforeach()

# Create PTX compilation target
add_custom_target(optix_kernels_ptx ALL
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/optix_kernels.ptx
)

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/optix_kernels.ptx
    COMMAND ${CMAKE_CUDA_COMPILER}
        -ptx
        -arch=sm_70
        -I${CMAKE_CURRENT_SOURCE_DIR}/include
        -I${OptiX_INCLUDE_DIR}
        -I${CUDAToolkit_INCLUDE_DIRS}
        --use_fast_math
        -O3
        ${CUDA_KERNEL_COMBINED}
        -o ${CMAKE_CURRENT_BINARY_DIR}/optix_kernels.ptx
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/raygen.cu
            ${CMAKE_CURRENT_SOURCE_DIR}/src/closesthit.cu
            ${CMAKE_CURRENT_SOURCE_DIR}/src/miss.cu
    COMMENT "Compiling CUDA kernels to PTX"
)

# Main library
add_library(lidar_simulator SHARED
    src/LidarSimulator.cpp
)

target_link_libraries(lidar_simulator
    CUDA::cuda_driver
    CUDA::cudart
    dl
)

# Python binding module
find_package(Python3 COMPONENTS Interpreter Development QUIET)

if(Python3_FOUND)
    # Try to find pybind11
    find_package(pybind11 CONFIG QUIET)
    
    if(pybind11_FOUND)
        message(STATUS "Building Python bindings with pybind11")
        
        pybind11_add_module(_optix_lidar_raycast
            src/python_binding.cpp
            src/LidarSimulator.cpp
        )
        
        target_link_libraries(_optix_lidar_raycast PRIVATE
            CUDA::cuda_driver
            CUDA::cudart
            dl
        )
        
        target_include_directories(_optix_lidar_raycast PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${OptiX_INCLUDE_DIR}
        )
        
        # Install the Python module
        install(TARGETS _optix_lidar_raycast
                LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX})
    else()
        message(STATUS "pybind11 not found, skipping Python bindings")
        message(STATUS "Install with: pip install pybind11")
    endif()
else()
    message(STATUS "Python3 not found, skipping Python bindings")
endif()

# Install PTX file
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/optix_kernels.ptx
        DESTINATION ${CMAKE_INSTALL_PREFIX})

# Install library
install(TARGETS lidar_simulator
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX})
