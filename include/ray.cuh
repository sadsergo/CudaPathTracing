#ifndef RAYH
#define RAYH

#include "vec3.cuh"
#include <cuda_runtime.h>

class ray
{
public:
  __device__ ray() {}
  __device__ ray(const vec3 &orig, const vec3 &dir) : orig(orig), dir(dir) {} 
  __device__ vec3 origin() const       { return orig; }
  __device__ vec3 direction() const    { return dir; }
  __device__ vec3 at(float t) const { return orig + t*dir; }

  vec3 orig;
  vec3 dir;
};

#endif