#ifndef CAMERAH
#define CAMERAH

#include "vec3.cuh"

class camera
{
public:
  __host__ __device__ camera() {}
  __host__ __device__ camera(const vec3 &up, const vec3 &pos,
                             const vec3 &target, const uint32_t width,
                             const uint32_t height, float fov)
      : fov(fov), AR(float(width) / float(height)), pos(pos)
  {
      dir = target - pos;
      dir.make_unit_vector();

      right = cross(dir, up);
      right.make_unit_vector();

      this->up = cross(dir, right);
      (this->up).make_unit_vector();
  }

  vec3 up;
  vec3 dir;
  vec3 pos;
  vec3 right;
  float fov;
  float AR;
};

#endif