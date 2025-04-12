
#include "kernels.cuh"
#include <iostream>

#define STBI_INCLUDE_STB_IMAGE_H
#include "stb_image.h"

int main()
{
  int nx = 1000, ny = 600;
  int tx = 8, ty = 8;

  std::cerr << "Rendering a " << nx << "x" << ny << " image ";
  std::cerr << "in " << tx << "x" << ty << " blocks.\n";

  render(nx, ny, tx, ty);

  return 0;
}