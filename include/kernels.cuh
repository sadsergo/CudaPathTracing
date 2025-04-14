#pragma once

#include <iostream>
#include <time.h>
#include <cstdint>

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "vec3.cuh"
#include "ray.cuh"
#include "camera.cuh"

#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)
void check_cuda(cudaError_t result, char const *const func,
                const char *const file, int const line);

__global__ void kernel_render(vec3 *fb, int max_x, int max_y, int nx, camera cam, curandState *rand_state);
__device__ vec3 color(const ray& r);

void render(vec3 *fb, int nx, int ny, int tx, int ty, int ns, camera cam);