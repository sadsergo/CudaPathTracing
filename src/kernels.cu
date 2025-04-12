
#include "kernels.cuh"


void check_cuda(cudaError_t result, char const *const func,
                const char *const file, int const line) {
  if (result) {
    std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at "
              << file << ":" << line << " '" << func << "' \n";
    // Make sure we call CUDA Device Reset before exiting
    cudaDeviceReset();
    exit(99);
  }
}

__device__ bool hit_sphere(const vec3& center, float radius, const ray& r) 
{
  vec3 oc = r.origin() - center;
  float a = dot(r.direction(), r.direction());
  float b = 2.0f * dot(oc, r.direction());
  float c = dot(oc, oc) - radius*radius;
  float discriminant = b*b - 4.0f*a*c;
  return (discriminant > 0.0f);
}

__device__ vec3 color(const ray& r) 
{
  if (hit_sphere(vec3(0,0,-1), 0.5, r))
        return vec3(1,0,0);
  vec3 unit_direction = unit_vector(r.direction());
  float t = 0.5f*(unit_direction.y() + 1.0f);
  return (1.0f-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
}

__global__ void kernel_render(vec3 *fb, int max_x, int max_y, camera &cam) 
{
  int j = threadIdx.x + blockIdx.x * blockDim.x;
  int i = threadIdx.y + blockIdx.y * blockDim.y;

  if ((i >= max_y) || (j >= max_x))
  {
    return;
  }

  vec3 P(j, i, 1);
  P /= vec3(max_x, max_y, 1);
  P = 2 * P - vec3(1, 1, 1);

  vec3 orig = cam.pos;
  vec3 dir = cam.dir + cam.right * P.x() * std::tan(cam.fov / 2.f) * cam.AR + cam.up * P.y() * std::tan(cam.fov / 2.f);
  dir.make_unit_vector();

  int pixel_index = i * max_x + j;

  ray r(orig, dir);
  // fb[pixel_index] = color(r);
  fb[pixel_index] = vec3(1,1,1);
}

void render(int nx, int ny, int tx, int ty, camera &cam) 
{
  long long num_pixels = nx * ny;
  size_t fb_size = num_pixels * sizeof(vec3);

  vec3 *fb;
  checkCudaErrors(cudaMallocManaged((void**)&fb, fb_size));

  dim3 blocks(nx / tx + 1, ny / ty + 1);
  dim3 threads(tx, ty);

  kernel_render<<<blocks, threads>>>(fb, nx, ny, cam);

  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  // Output FB as Image
  std::cout << "P3\n" << nx << " " << ny << "\n255\n";
  for (int i = ny-1; i >= 0; i--) {
      for (int j = 0; j < nx; j++) {
          size_t pixel_index = i*nx + j;

          int ir = int(255.99*fb[pixel_index].r());
          int ig = int(255.99*fb[pixel_index].g());
          int ib = int(255.99*fb[pixel_index].b());
          std::cout << ir << " " << ig << " " << ib << "\n";
      }
  }

  checkCudaErrors(cudaFree(fb));
}