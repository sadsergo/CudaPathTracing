
#include "kernels.cuh"
#include <iostream>

#define STBI_INCLUDE_STB_IMAGE_H
#include "stb_image.h"

int main()
{
  int nx = 1000, ny = 600;
  int tx = 8, ty = 8;
  int ns = 100;

  camera cam(vec3(2, 0, 0), vec3(0, 0, 0), vec3(0, 1, 0), nx, ny, 3.14f / 4.f);

  std::cerr << "Rendering a " << nx << "x" << ny << " image ";
  std::cerr << "in " << tx << "x" << ty << " blocks.\n";

  render(nx, ny, tx, ty, ns, cam);

  return 0;
}