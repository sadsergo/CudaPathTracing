#pragma once

#include <iostream>
#include <time.h>

#include <cuda_runtime.h>

#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)
void check_cuda(cudaError_t result, char const *const func,
                const char *const file, int const line);

__global__ void render(float *fb, int max_x, int max_y);