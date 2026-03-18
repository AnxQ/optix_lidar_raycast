#include <optix.h>
#include "LaunchParams.h"

// params is defined in raygen.cu and will be available after merging

extern "C" __global__ void __anyhit__lidar() {
    const unsigned int prim_idx = optixGetPrimitiveIndex();
    const unsigned int hit_batch_idx = prim_idx / static_cast<unsigned int>(params.num_triangles);
    const unsigned int expected_batch_idx = optixGetPayload_5();

    if (hit_batch_idx != expected_batch_idx) {
        optixIgnoreIntersection();
    }
}

extern "C" __global__ void __closesthit__lidar() {
    // Get hit information
    const float3 ray_orig = optixGetWorldRayOrigin();
    const float3 ray_dir = optixGetWorldRayDirection();
    const float t_hit = optixGetRayTmax();
    
    // Compute hit point
    float3 hit_point;
    hit_point.x = ray_orig.x + t_hit * ray_dir.x;
    hit_point.y = ray_orig.y + t_hit * ray_dir.y;
    hit_point.z = ray_orig.z + t_hit * ray_dir.z;
    
    // Get primitive (triangle) index
    const unsigned int prim_idx = optixGetPrimitiveIndex();
    
    // Convert expanded triangle index back to original face_id
    // Since we replicated triangles for each batch, we need to modulo
    const unsigned int face_id = prim_idx % params.num_triangles;
    
    // Encode hit information into payload
    // We use 5 unsigned int registers:
    // p0: hit_point.x
    // p1: hit_point.y
    // p2: hit_point.z
    // p3: distance
    // p4: face_id
    
    unsigned int p0 = __float_as_uint(hit_point.x);
    unsigned int p1 = __float_as_uint(hit_point.y);
    unsigned int p2 = __float_as_uint(hit_point.z);
    unsigned int p3 = __float_as_uint(t_hit);
    unsigned int p4 = face_id;
    
    optixSetPayload_0(p0);
    optixSetPayload_1(p1);
    optixSetPayload_2(p2);
    optixSetPayload_3(p3);
    optixSetPayload_4(p4);
}
