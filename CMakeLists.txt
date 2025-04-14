cmake_minimum_required(VERSION 3.18)  # 3.18+ is better for Ada GPUs
project(engine LANGUAGES CXX CUDA)

# Set your source files
set(SOURCES
    src/main.cpp
    src/kernels.cu

)

set(CMAKE_CUDA_HOST_COMPILER clang-14)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=compute_61")  # Pascal (GTX 10-series)

# Create the executable
add_executable(engine ${SOURCES})

target_include_directories(engine PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Target Ada architecture (RTX 4080 is compute capability 8.9)
set_target_properties(engine PROPERTIES
    CUDA_ARCHITECTURES 89
)

# Optional: nvcc flags
target_compile_options(engine PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)