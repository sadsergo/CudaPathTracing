
#include "kernels.cuh"
#include <vector>

// stb_image is a single-header C library, which means one of your cpp files must have
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"

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
  if (hit_sphere(vec3(0,0,0), 0.5, r))
        return vec3(1,0,0);
  vec3 unit_direction = unit_vector(r.direction());
  float t = 0.5f*(unit_direction.y() + 1.0f);
  return (1.0f-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
}

__global__ void kernel_render_init(int max_x, int max_y, curandState *rand_state) 
{
  int j = threadIdx.x + blockIdx.x * blockDim.x;
  int i = threadIdx.y + blockIdx.y * blockDim.y;
  if((j >= max_x) || (i >= max_y)) return;
  int pixel_index = i*max_x + j;
  //Each thread gets same seed, a different sequence number, no offset
  curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}

__global__ void kernel_render(vec3 *fb, int max_x, int max_y, int ns, camera cam, curandState *rand_state) 
{
  int j = threadIdx.x + blockIdx.x * blockDim.x;
  int i = threadIdx.y + blockIdx.y * blockDim.y;

  if ((i >= max_y) || (j >= max_x))
  {
    return;
  }

  int pixel_index = i * max_x + j;
  curandState local_rand_state = rand_state[pixel_index];

  for (int s = 0; s < ns; s++)
  {
    vec3 P((float)j + curand_uniform(&local_rand_state), (float)i + curand_uniform(&local_rand_state), 1.f);
    P /= vec3(max_x, max_y, 1.f);
    P = 2 * P - vec3(1.f, 1.f, 1.f);

    vec3 orig = cam.pos;
    vec3 dir = unit_vector(cam.dir + cam.right * P.x() * std::tan(cam.fov / 2.f) * cam.AR + cam.up * P.y() * std::tan(cam.fov / 2.f));

    ray r(orig, dir);
    fb[pixel_index] += color(r);
  }

  fb[pixel_index] /= (float)ns;
}

void render(vec3 *fb, int nx, int ny, int tx, int ty, int ns, camera cam)
{
  long long num_pixels = nx * ny;

  // allocate random state
  curandState *d_rand_state;
  checkCudaErrors(cudaMalloc((void **)&d_rand_state, num_pixels*sizeof(curandState)));

  dim3 blocks(nx / tx + 1, ny / ty + 1);
  dim3 threads(tx, ty);

  kernel_render_init<<<blocks, threads>>>(nx, ny, d_rand_state);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  kernel_render<<<blocks, threads>>>(fb, nx, ny, ns, cam, d_rand_state);

  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  // Output FB as Image
  std::vector<unsigned char> rgb_buffer(nx * ny * 3);

  for (int i = 0; i < ny; i++)
  {
    for (int j = 0; j < nx; j++)
    {
      size_t im_index = 3 * i * nx + j * 3;
      size_t pixel_index = i * nx + j;
      rgb_buffer[im_index + 0] = static_cast<unsigned char>(fb[pixel_index].x() * 255.0f);     // R
      rgb_buffer[im_index + 1] = static_cast<unsigned char>(fb[pixel_index].y() * 255.0f);     // G
      rgb_buffer[im_index + 2] = static_cast<unsigned char>(fb[pixel_index].z() * 255.0f);     // B
    }
  }

  stbi_write_png("out.png", nx, ny, 3, rgb_buffer.data(), nx * 3);

  checkCudaErrors(cudaFree(d_rand_state));
}