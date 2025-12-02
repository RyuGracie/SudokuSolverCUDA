cmake_minimum_required(VERSION 3.18)

project(MyCudaProject LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Source directory
set(SRC_DIR "${CMAKE_SOURCE_DIR}/src")

# ---------------------------
# CPU executable
# ---------------------------
add_executable(main_cpu
    ${SRC_DIR}/main_cpu.cpp
)

# Place CPU executable in build/bin/
set_target_properties(main_cpu PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# ---------------------------
# GPU executable
# ---------------------------
add_executable(main_gpu
    ${SRC_DIR}/main.cu
)

# Place GPU executable in build/bin/
set_target_properties(main_gpu PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    CUDA_SEPARABLE_COMPILATION ON
)

# Optional: CUDA architecture (uncomment to set manually)
# set_target_properties(main_gpu PROPERTIES CUDA_ARCHITECTURES "75")
