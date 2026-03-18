#pragma once

#include <optix.h>
#include <cuda_runtime.h>

struct LaunchParams {
    // Ray tracing output
    float3* hit_points;      // (batch_size * num_rays, 3) - hit positions
    float* hit_distances;    // (batch_size * num_rays) - ray distances
    int* hit_face_ids;       // (batch_size * num_rays) - triangle indices (-1 if miss)
    
    // Mesh data (batch support)
    float3* vertices;        // (batch_size * num_vertices, 3) - vertex positions
    int3* triangles;         // (num_triangles, 3) - triangle vertex indices
    int num_vertices;        // number of vertices per mesh
    int num_triangles;       // number of triangles per mesh
    
    // LiDAR configuration
    float3* lidar_positions;    // (batch_size, 3) - sensor origins
    float3* lidar_orientations; // (batch_size, 3) - Euler angles (roll, pitch, yaw)
    int num_rays;               // total rays per scan
    int num_vertical_beams;     // vertical resolution
    int num_horizontal_samples; // horizontal resolution
    float vertical_fov_min;     // vertical FOV range (radians)
    float vertical_fov_max;
    float horizontal_fov_min;   // horizontal FOV range (radians)
    float horizontal_fov_max;
    float max_range;            // maximum detection range
    
    // Runtime parameters
    int batch_size;
    OptixTraversableHandle traversable_handle; // Acceleration structure handle
};

// Helper structure for ray payload
struct RayPayload {
    float3 hit_point;
    float distance;
    int face_id;
    bool hit;
};
