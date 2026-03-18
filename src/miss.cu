#include <optix.h>
#include "LaunchParams.h"

extern "C" __global__ void __miss__lidar() {
    // Ray missed all geometry
    // Set hit point to origin (or far away point)
    const float3 ray_orig = optixGetWorldRayOrigin();
    const float3 ray_dir = optixGetWorldRayDirection();
    const float max_t = optixGetRayTmax();
    
    // Set hit point to maximum distance
    float3 miss_point;
    miss_point.x = ray_orig.x + max_t * ray_dir.x;
    miss_point.y = ray_orig.y + max_t * ray_dir.y;
    miss_point.z = ray_orig.z + max_t * ray_dir.z;
    
    // Encode miss information
    unsigned int p0 = __float_as_uint(miss_point.x);
    unsigned int p1 = __float_as_uint(miss_point.y);
    unsigned int p2 = __float_as_uint(miss_point.z);
    unsigned int p3 = __float_as_uint(max_t);
    unsigned int p4 = 0xFFFFFFFF;  // Use -1 as miss marker
    
    optixSetPayload_0(p0);
    optixSetPayload_1(p1);
    optixSetPayload_2(p2);
    optixSetPayload_3(p3);
    optixSetPayload_4(p4);
}
