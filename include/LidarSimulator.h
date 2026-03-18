#pragma once

#include <optix.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>
#include <mutex>
#include <vector>
#include <string>

class LidarSimulator {
public:
    LidarSimulator();
    ~LidarSimulator();
    
    // Initialize OptiX context and pipeline
    // device_id: CUDA device ID to use (-1 = use current device)
    void initialize(int device_id = -1);
    
    // Set LiDAR configuration
    void setLidarConfig(
        int num_vertical_beams,
        int num_horizontal_samples,
        float vertical_fov_min,    // degrees
        float vertical_fov_max,
        float horizontal_fov_min,
        float horizontal_fov_max,
        float max_range
    );
    
    // Set mesh data for batch processing
    // vertices: (batch_size, num_vertices, 3) flattened
    // triangles: (num_triangles, 3) - shared across batch
    void setMeshBatch(
        const float* vertices,
        const int* triangles,
        int batch_size,
        int num_vertices,
        int num_triangles,
        bool vertices_on_device = false,
        bool triangles_on_device = false
    );
    
    // Perform raycasting for all meshes in batch
    // lidar_positions: (batch_size, 3) - sensor positions
    // lidar_orientations: (batch_size, 3) - Euler angles (roll, pitch, yaw) in degrees
    // output_points: (batch_size, num_rays, 3) - hit points
    // output_distances: (batch_size, num_rays) - distances
    // output_face_ids: (batch_size, num_rays) - face indices (-1 if miss)
    void raycastBatch(
        const float* lidar_positions,
        const float* lidar_orientations,
        float* output_points,
        float* output_distances,
        int* output_face_ids,
        bool input_on_device = false,
        bool output_on_device = false
    );
    
    // Cleanup resources
    void cleanup();
    
    // Get number of rays per scan
    int getNumRays() const { return num_rays_; }
    
private:
    // OptiX objects
    OptixDeviceContext context_ = nullptr;
    OptixModule module_ = nullptr;
    OptixPipeline pipeline_ = nullptr;
    OptixShaderBindingTable sbt_ = {};
    
    // Program groups
    OptixProgramGroup raygen_pg_ = nullptr;
    OptixProgramGroup miss_pg_ = nullptr;
    OptixProgramGroup hitgroup_pg_ = nullptr;
    OptixProgramGroup anyhit_pg_ = nullptr;
    
    // CUDA stream
    CUstream stream_ = nullptr;
    
    // Device memory
    CUdeviceptr d_gas_output_buffer_ = 0;    // Acceleration structure buffer
    CUdeviceptr d_vertices_ = 0;             // Vertex buffer
    CUdeviceptr d_triangles_ = 0;            // Triangle buffer
    CUdeviceptr d_lidar_positions_ = 0;      // Sensor positions
    CUdeviceptr d_lidar_orientations_ = 0;   // Sensor orientations
    CUdeviceptr d_hit_points_ = 0;           // Output hit points
    CUdeviceptr d_hit_distances_ = 0;        // Output distances
    CUdeviceptr d_hit_face_ids_ = 0;         // Output face IDs
    CUdeviceptr d_launch_params_ = 0;        // Launch parameters
    
    // Acceleration structure handles (one per batch item)
    std::vector<OptixTraversableHandle> gas_handles_;
    OptixTraversableHandle ias_handle_ = 0;  // Instance AS handle
    CUdeviceptr d_ias_output_buffer_ = 0;
    
    // Configuration
    int num_vertical_beams_ = 32;
    int num_horizontal_samples_ = 1024;
    int num_rays_ = 0;
    float vertical_fov_min_ = -0.2793f;   // -16 degrees
    float vertical_fov_max_ = 0.2793f;    // 16 degrees
    float horizontal_fov_min_ = 0.0f;
    float horizontal_fov_max_ = 6.2832f;  // 360 degrees
    float max_range_ = 100.0f;
    
    // Batch state
    int batch_size_ = 0;
    int num_vertices_ = 0;
    int num_triangles_ = 0;

    // Runtime safety
    std::mutex raycast_mutex_;
    
    // Helper methods
    void createContext();
    void createModule();
    void createProgramGroups(OptixProgramGroup* raygen_pg,
                            OptixProgramGroup* miss_pg,
                            OptixProgramGroup* hitgroup_pg,
                            OptixProgramGroup* anyhit_pg);
    void createPipeline(OptixProgramGroup raygen_pg,
                       OptixProgramGroup miss_pg,
                       OptixProgramGroup hitgroup_pg);
    void createSBT(OptixProgramGroup raygen_pg,
                  OptixProgramGroup miss_pg,
                  OptixProgramGroup hitgroup_pg);
    void buildAccelerationStructure();
    void updateAccelerationStructure();
    
    // Utility
    void allocateBuffers();
    void freeBuffers();
    static void contextLogCallback(unsigned int level, const char* tag,
                                   const char* message, void* cbdata);
};
