#include <optix.h>
#include <math_constants.h>
#include "LaunchParams.h"

extern "C" {
    __constant__ LaunchParams params;
}

// Compute ray direction based on LiDAR scan pattern
__device__ float3 computeRayDirection(
    int ray_idx,
    int batch_idx,
    int num_vertical_beams,
    int num_horizontal_samples,
    float vertical_fov_min,
    float vertical_fov_max,
    float horizontal_fov_min,
    float horizontal_fov_max,
    float3 orientation) {
    
    // Decompose ray index
    int beam_idx = ray_idx / num_horizontal_samples;
    int sample_idx = ray_idx % num_horizontal_samples;
    
    // Compute vertical angle (elevation)
    float vertical_t = (float)beam_idx / (float)(num_vertical_beams - 1);
    float elevation = vertical_fov_min + vertical_t * (vertical_fov_max - vertical_fov_min);
    
    // Compute horizontal angle (azimuth)
    float horizontal_t = (float)sample_idx / (float)(num_horizontal_samples - 1);
    float azimuth = horizontal_fov_min + horizontal_t * (horizontal_fov_max - horizontal_fov_min);
    
    // Compute direction in sensor frame (assuming +X forward, +Y left, +Z up)
    float cos_elev = cosf(elevation);
    float sin_elev = sinf(elevation);
    float cos_azim = cosf(azimuth);
    float sin_azim = sinf(azimuth);
    
    float3 dir_local;
    dir_local.x = cos_elev * cos_azim;  // Forward
    dir_local.y = cos_elev * sin_azim;  // Left
    dir_local.z = sin_elev;              // Up
    
    // Apply sensor orientation (Euler angles: roll, pitch, yaw)
    // For simplicity, we assume roll=0 and only apply pitch and yaw
    const float deg2rad = CUDART_PI_F / 180.0f;
    float roll = orientation.x * deg2rad;
    float pitch = orientation.y * deg2rad;
    float yaw = orientation.z * deg2rad;
    
    // Rotation matrices (ZYX convention)
    float cr = cosf(roll);
    float sr = sinf(roll);
    float cp = cosf(pitch);
    float sp = sinf(pitch);
    float cy = cosf(yaw);
    float sy = sinf(yaw);
    
    // Combined rotation matrix (Rz * Ry * Rx)
    float3 dir_world;
    dir_world.x = (cy * cp) * dir_local.x + 
                  (cy * sp * sr - sy * cr) * dir_local.y + 
                  (cy * sp * cr + sy * sr) * dir_local.z;
    
    dir_world.y = (sy * cp) * dir_local.x + 
                  (sy * sp * sr + cy * cr) * dir_local.y + 
                  (sy * sp * cr - cy * sr) * dir_local.z;
    
    dir_world.z = (-sp) * dir_local.x + 
                  (cp * sr) * dir_local.y + 
                  (cp * cr) * dir_local.z;
    
    // Normalize
    float len = sqrtf(dir_world.x * dir_world.x + 
                     dir_world.y * dir_world.y + 
                     dir_world.z * dir_world.z);
    dir_world.x /= len;
    dir_world.y /= len;
    dir_world.z /= len;
    
    return dir_world;
}

extern "C" __global__ void __raygen__lidar_scan() {
    const int ray_idx = static_cast<int>(optixGetLaunchIndex().x);
    const int batch_idx = static_cast<int>(optixGetLaunchIndex().y);

    if (batch_idx >= params.batch_size || ray_idx >= params.num_rays) return;
    
    // Get LiDAR pose for this batch
    float3 origin = params.lidar_positions[batch_idx];
    float3 orientation = params.lidar_orientations[batch_idx];
    
    // Compute ray direction
    float3 direction = computeRayDirection(
        ray_idx,
        batch_idx,
        params.num_vertical_beams,
        params.num_horizontal_samples,
        params.vertical_fov_min,
        params.vertical_fov_max,
        params.horizontal_fov_min,
        params.horizontal_fov_max,
        orientation
    );
    
    // Trace ray
    unsigned int p0, p1, p2, p3, p4;
    unsigned int p5 = static_cast<unsigned int>(batch_idx);
    
    optixTrace(
        params.traversable_handle,
        origin,
        direction,
        0.0f,                    // tmin
        params.max_range,        // tmax
        0.0f,                    // rayTime
        OptixVisibilityMask(1),
        OPTIX_RAY_FLAG_NONE,
        0,                       // SBT offset
        1,                       // SBT stride
        0,                       // missSBTIndex
        p0, p1, p2, p3, p4, p5   // Payload
    );
    
    // Decode payload
    float3 hit_point;
    hit_point.x = __uint_as_float(p0);
    hit_point.y = __uint_as_float(p1);
    hit_point.z = __uint_as_float(p2);
    
    float distance = __uint_as_float(p3);
    int face_id = (int)p4;
    
    // For miss, face_id will be 0xFFFFFFFF
    if (p4 == 0xFFFFFFFF) {
        face_id = -1;
    }
    
    // Write output
    const int output_idx = batch_idx * params.num_rays + ray_idx;
    params.hit_points[output_idx] = hit_point;
    params.hit_distances[output_idx] = distance;
    params.hit_face_ids[output_idx] = face_id;
}
