
#include "kernels.cuh"
#include <iostream>

int main()
{
  int nx = 1000, ny = 600;
  int tx = 8, ty = 8;
  int ns = 100;

  vec3 *fb;
  size_t fb_size = nx * ny * sizeof(vec3);

  checkCudaErrors(cudaMallocManaged((void**)&fb, fb_size));

  camera cam(vec3(2, 0, 0), vec3(0, 0, 0), vec3(0, 1, 0), nx, ny, 3.14f / 4.f);

  std::cerr << "Rendering a " << nx << "x" << ny << " image ";
  std::cerr << "in " << tx << "x" << ty << " blocks.\n";

  render(fb, nx, ny, tx, ty, ns, cam);

  checkCudaErrors(cudaFree(fb));

  return 0;
}