#include "LidarSimulator.h"
#include "LaunchParams.h"
#include <optix_function_table_definition.h>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <dlfcn.h>

// SBT record structures (must match alignment requirements)
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) RaygenRecord {
    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) MissRecord {
    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) HitgroupRecord {
    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

#define CUDA_CHECK(call)                                                       \
    {                                                                          \
        cudaError_t error = call;                                              \
        if (error != cudaSuccess) {                                            \
            throw std::runtime_error(std::string("CUDA error: ") +            \
                                   cudaGetErrorString(error) +                 \
                                   " at " + __FILE__ + ":" +                   \
                                   std::to_string(__LINE__));                  \
        }                                                                      \
    }

#define OPTIX_CHECK(call)                                                      \
    {                                                                          \
        OptixResult res = call;                                                \
        if (res != OPTIX_SUCCESS) {                                            \
            throw std::runtime_error(std::string("OptiX error: ") +           \
                                   optixGetErrorName(res) +                    \
                                   " at " + __FILE__ + ":" +                   \
                                   std::to_string(__LINE__));                  \
        }                                                                      \
    }

// External CUDA code (will be compiled to PTX)
extern "C" const char embedded_ptx_code[];

LidarSimulator::LidarSimulator() {
}

LidarSimulator::~LidarSimulator() {
    cleanup();
}

void LidarSimulator::contextLogCallback(unsigned int level, const char* tag,
                                       const char* message, void* cbdata) {
    // Suppress OptiX logs
    // Uncomment the line below to enable logging:
    // std::cerr << "[OptiX][" << level << "][" << tag << "]: " << message << std::endl;
}

void LidarSimulator::initialize(int device_id) {
    // Try to initialize CUDA carefully to handle fork scenarios
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    
    // If this fails, we might be in a forked process
    if (err != cudaSuccess) {
        // Try to reset the error and proceed
        cudaGetLastError(); // Clear the error
        
        // In a forked process, we need to be very careful
        // Try to detect if we're in a forked child process
        // If cudaGetDeviceCount fails, it might mean CUDA was already initialized in parent
        
        // Attempt to get device count again after clearing error
        err = cudaGetDeviceCount(&deviceCount);
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("Failed to initialize CUDA. ") +
                std::string("If using DataLoader with num_workers > 0, ") +
                std::string("set multiprocessing_context='spawn' or use num_workers=0. ") +
                std::string("Error: ") + cudaGetErrorString(err)
            );
        }
    }
    
    if (deviceCount == 0) {
        throw std::runtime_error("No CUDA devices available");
    }
    
    // Set CUDA device
    if (device_id >= 0) {
        if (device_id >= deviceCount) {
            throw std::runtime_error("Invalid device ID: " + std::to_string(device_id) + 
                                   " (available devices: 0-" + std::to_string(deviceCount-1) + ")");
        }
        err = cudaSetDevice(device_id);
        if (err != cudaSuccess) {
            cudaGetLastError(); // Clear error
            throw std::runtime_error(
                std::string("Cannot set CUDA device in forked process. ") +
                std::string("Use DataLoader with num_workers=0 or multiprocessing_context='spawn'")
            );
        }
    } else {
        // Try to get current device
        int current_device = 0;
        err = cudaGetDevice(&current_device);
        if (err != cudaSuccess) {
            cudaGetLastError(); // Clear error
            // Try to set device 0
            err = cudaSetDevice(0);
            if (err != cudaSuccess) {
                cudaGetLastError(); // Clear error
                throw std::runtime_error(
                    std::string("Cannot initialize CUDA in forked process. ") +
                    std::string("Use DataLoader with num_workers=0 or multiprocessing_context='spawn'")
                );
            }
            device_id = 0;
        } else {
            device_id = current_device;
        }
    }
    
    // Try to synchronize - this will fail in forked processes
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cudaGetLastError(); // Clear error
        throw std::runtime_error(
            std::string("Cannot use CUDA in forked subprocess. ") +
            std::string("Solutions:\n") +
            std::string("  1. Set DataLoader num_workers=0\n") +
            std::string("  2. Use multiprocessing_context='spawn' in DataLoader\n") +
            std::string("  3. Set persistent_workers=True in DataLoader\n") +
            std::string("Error: ") + cudaGetErrorString(err)
        );
    }
    
    // Initialize OptiX
    OPTIX_CHECK(optixInit());
    
    // Create context
    createContext();
    
    // Create module from PTX
    createModule();
    
    // Create program groups
    createProgramGroups(&raygen_pg_, &miss_pg_, &hitgroup_pg_, &anyhit_pg_);
    
    // Create pipeline
    createPipeline(raygen_pg_, miss_pg_, hitgroup_pg_);
    
    // Create shader binding table
    createSBT(raygen_pg_, miss_pg_, hitgroup_pg_);
    
    // Create CUDA stream
    CUDA_CHECK(cudaStreamCreate(&stream_));
    
    std::cout << "LidarSimulator initialized successfully" << std::endl;
}

void LidarSimulator::createContext() {
    // Get CUDA device
    CUcontext cu_ctx = 0;  // Use current context
    
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = &LidarSimulator::contextLogCallback;
    options.logCallbackLevel = 0;  // 0 = Disable logs, 4 = Print all messages
    
    OPTIX_CHECK(optixDeviceContextCreate(cu_ctx, &options, &context_));
}

void LidarSimulator::createModule() {
    // Module compilation options
    OptixModuleCompileOptions module_compile_options = {};
    module_compile_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;
    
    // Pipeline compilation options
    OptixPipelineCompileOptions pipeline_compile_options = {};
    pipeline_compile_options.usesMotionBlur = false;
    pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipeline_compile_options.numPayloadValues = 6;  // 5 outputs + expected batch id for any-hit filtering
    pipeline_compile_options.numAttributeValues = 2;  // Standard triangle attributes
    pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
    
    // Load PTX code. Prefer the PTX shipped next to the loaded extension module
    // to avoid accidentally picking stale repository artifacts.
    std::vector<std::string> ptx_paths;
    Dl_info dl_info = {};
    if (dladdr(reinterpret_cast<void*>(&LidarSimulator::contextLogCallback), &dl_info) != 0 && dl_info.dli_fname) {
        std::string module_path(dl_info.dli_fname);
        std::size_t slash_pos = module_path.find_last_of('/');
        if (slash_pos != std::string::npos) {
            ptx_paths.push_back(module_path.substr(0, slash_pos + 1) + "optix_kernels.ptx");
        }
    }

    // Fallback search paths for local/dev workflows.
    ptx_paths.push_back("optix_kernels.ptx");
    ptx_paths.push_back("python/optix_kernels.ptx");
    ptx_paths.push_back("optix_lidar_raycast_src/python/optix_kernels.ptx");
    ptx_paths.push_back("../optix_lidar_raycast_src/python/optix_kernels.ptx");
    ptx_paths.push_back("../optix_lidar_raycast_src/build/optix_kernels.ptx");
    ptx_paths.push_back("optix_lidar_raycast/python/optix_kernels.ptx");
    ptx_paths.push_back("../optix_lidar_raycast/python/optix_kernels.ptx");
    ptx_paths.push_back("../optix_lidar_raycast/build/optix_kernels.ptx");
    ptx_paths.push_back("build/optix_kernels.ptx");
    
    std::string ptx_filename;
    std::ifstream ptx_file;
    bool found = false;
    
    for (const auto& path : ptx_paths) {
        ptx_file.open(path);
        if (ptx_file.good()) {
            ptx_filename = path;
            found = true;
            break;
        }
        ptx_file.close();
    }
    
    if (!found) {
        std::string error_msg = "Could not open PTX file. Tried:\n";
        for (const auto& path : ptx_paths) {
            error_msg += "  - " + path + "\n";
        }
        throw std::runtime_error(error_msg);
    }
    
    std::string ptx_code((std::istreambuf_iterator<char>(ptx_file)),
                         std::istreambuf_iterator<char>());
    
    char log[2048];
    size_t sizeof_log = sizeof(log);
    
    OPTIX_CHECK(optixModuleCreate(
        context_,
        &module_compile_options,
        &pipeline_compile_options,
        ptx_code.c_str(),
        ptx_code.size(),
        log,
        &sizeof_log,
        &module_
    ));
    
    if (sizeof_log > 1) {
        std::cout << "Module creation log:\n" << log << std::endl;
    }
}

void LidarSimulator::createProgramGroups(OptixProgramGroup* raygen_pg,
                                        OptixProgramGroup* miss_pg,
                                        OptixProgramGroup* hitgroup_pg,
                                        OptixProgramGroup* anyhit_pg) {
    char log[2048];
    size_t sizeof_log = sizeof(log);
    
    OptixProgramGroupOptions program_group_options = {};
    
    // Ray generation program
    OptixProgramGroupDesc raygen_desc = {};
    raygen_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygen_desc.raygen.module = module_;
    raygen_desc.raygen.entryFunctionName = "__raygen__lidar_scan";
    
    OPTIX_CHECK(optixProgramGroupCreate(
        context_,
        &raygen_desc,
        1,
        &program_group_options,
        log,
        &sizeof_log,
        raygen_pg
    ));
    
    // Miss program
    OptixProgramGroupDesc miss_desc = {};
    miss_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_desc.miss.module = module_;
    miss_desc.miss.entryFunctionName = "__miss__lidar";
    
    OPTIX_CHECK(optixProgramGroupCreate(
        context_,
        &miss_desc,
        1,
        &program_group_options,
        log,
        &sizeof_log,
        miss_pg
    ));
    
    // Hit group program
    OptixProgramGroupDesc hitgroup_desc = {};
    hitgroup_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitgroup_desc.hitgroup.moduleCH = module_;
    hitgroup_desc.hitgroup.entryFunctionNameCH = "__closesthit__lidar";
    hitgroup_desc.hitgroup.moduleAH = module_;
    hitgroup_desc.hitgroup.entryFunctionNameAH = "__anyhit__lidar";
    
    OPTIX_CHECK(optixProgramGroupCreate(
        context_,
        &hitgroup_desc,
        1,
        &program_group_options,
        log,
        &sizeof_log,
        hitgroup_pg
    ));

    *anyhit_pg = *hitgroup_pg;
}

void LidarSimulator::createPipeline(OptixProgramGroup raygen_pg,
                                   OptixProgramGroup miss_pg,
                                   OptixProgramGroup hitgroup_pg) {
    OptixProgramGroup program_groups[] = {raygen_pg, miss_pg, hitgroup_pg};
    
    // Need to provide pipeline compile options
    OptixPipelineCompileOptions pipeline_compile_options = {};
    pipeline_compile_options.usesMotionBlur = false;
    pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipeline_compile_options.numPayloadValues = 6;  // 5 outputs + expected batch id for any-hit filtering
    pipeline_compile_options.numAttributeValues = 2;
    pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
    
    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth = 1;
    
    char log[2048];
    size_t sizeof_log = sizeof(log);
    
    OPTIX_CHECK(optixPipelineCreate(
        context_,
        &pipeline_compile_options,  // Need to pass pipeline compile options
        &pipeline_link_options,
        program_groups,
        sizeof(program_groups) / sizeof(program_groups[0]),
        log,
        &sizeof_log,
        &pipeline_
    ));
    
    if (sizeof_log > 1) {
        std::cout << "Pipeline creation log:\n" << log << std::endl;
    }
    
    // Set stack sizes (using reasonable defaults)
    // Since we use OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS,
    // maxTraversableGraphDepth must be 1
    const unsigned int direct_callable_stack_size_from_traversal = 0;
    const unsigned int direct_callable_stack_size_from_state = 0;
    const unsigned int continuation_stack_size = 2048;
    const unsigned int max_traversable_graph_depth = 1;  // Must be 1 for single GAS
    
    OPTIX_CHECK(optixPipelineSetStackSize(
        pipeline_,
        direct_callable_stack_size_from_traversal,
        direct_callable_stack_size_from_state,
        continuation_stack_size,
        max_traversable_graph_depth
    ));
}

void LidarSimulator::createSBT(OptixProgramGroup raygen_pg,
                              OptixProgramGroup miss_pg,
                              OptixProgramGroup hitgroup_pg) {
    // Raygen record
    CUdeviceptr d_raygen_record;
    size_t raygen_record_size = sizeof(RaygenRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_raygen_record), raygen_record_size));
    
    RaygenRecord raygen_record;
    OPTIX_CHECK(optixSbtRecordPackHeader(raygen_pg, &raygen_record));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_raygen_record),
        &raygen_record,
        raygen_record_size,
        cudaMemcpyHostToDevice
    ));
    
    // Miss record
    CUdeviceptr d_miss_record;
    size_t miss_record_size = sizeof(MissRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_miss_record), miss_record_size));
    
    MissRecord miss_record;
    OPTIX_CHECK(optixSbtRecordPackHeader(miss_pg, &miss_record));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_miss_record),
        &miss_record,
        miss_record_size,
        cudaMemcpyHostToDevice
    ));
    
    // Hitgroup record
    CUdeviceptr d_hitgroup_record;
    size_t hitgroup_record_size = sizeof(HitgroupRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_hitgroup_record), hitgroup_record_size));
    
    HitgroupRecord hitgroup_record;
    OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_pg, &hitgroup_record));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_hitgroup_record),
        &hitgroup_record,
        hitgroup_record_size,
        cudaMemcpyHostToDevice
    ));
    
    // Setup SBT
    sbt_.raygenRecord = d_raygen_record;
    sbt_.missRecordBase = d_miss_record;
    sbt_.missRecordStrideInBytes = sizeof(MissRecord);
    sbt_.missRecordCount = 1;
    sbt_.hitgroupRecordBase = d_hitgroup_record;
    sbt_.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
    sbt_.hitgroupRecordCount = 1;
}

void LidarSimulator::setLidarConfig(
    int num_vertical_beams,
    int num_horizontal_samples,
    float vertical_fov_min,
    float vertical_fov_max,
    float horizontal_fov_min,
    float horizontal_fov_max,
    float max_range) {
    
    num_vertical_beams_ = num_vertical_beams;
    num_horizontal_samples_ = num_horizontal_samples;
    num_rays_ = num_vertical_beams * num_horizontal_samples;
    
    // Convert degrees to radians
    vertical_fov_min_ = vertical_fov_min * M_PI / 180.0f;
    vertical_fov_max_ = vertical_fov_max * M_PI / 180.0f;
    horizontal_fov_min_ = horizontal_fov_min * M_PI / 180.0f;
    horizontal_fov_max_ = horizontal_fov_max * M_PI / 180.0f;
    max_range_ = max_range;
    
    std::cout << "LiDAR config: " << num_vertical_beams << "x" << num_horizontal_samples
              << " = " << num_rays_ << " rays" << std::endl;
}

void LidarSimulator::setMeshBatch(
    const float* vertices,
    const int* triangles,
    int batch_size,
    int num_vertices,
    int num_triangles,
    bool vertices_on_device,
    bool triangles_on_device) {
    
    batch_size_ = batch_size;
    num_vertices_ = num_vertices;
    num_triangles_ = num_triangles;
    
    // Free old buffers if exists
    freeBuffers();
    
    // Allocate new buffers
    allocateBuffers();
    
    // Copy mesh data to device
    size_t vertex_buffer_size = batch_size * num_vertices * sizeof(float3);
    size_t triangle_buffer_size = num_triangles * sizeof(int3);
    cudaMemcpyKind vertex_copy_kind = vertices_on_device ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;
    cudaMemcpyKind triangle_copy_kind = triangles_on_device ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;
    
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_vertices_),
        vertices,
        vertex_buffer_size,
        vertex_copy_kind
    ));
    
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_triangles_),
        triangles,
        triangle_buffer_size,
        triangle_copy_kind
    ));
    
    // Build acceleration structure
    buildAccelerationStructure();
    
    // Verbose output disabled for production use
    // std::cout << "Mesh batch set: " << batch_size << " meshes, "
    //           << num_vertices << " vertices, " << num_triangles << " triangles" << std::endl;
}

void LidarSimulator::allocateBuffers() {
    // Mesh buffers
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_vertices_),
                         batch_size_ * num_vertices_ * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_triangles_),
                         num_triangles_ * sizeof(int3)));
    
    // LiDAR config buffers
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_lidar_positions_),
                         batch_size_ * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_lidar_orientations_),
                         batch_size_ * sizeof(float3)));
    
    // Output buffers
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_hit_points_),
                         batch_size_ * num_rays_ * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_hit_distances_),
                         batch_size_ * num_rays_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_hit_face_ids_),
                         batch_size_ * num_rays_ * sizeof(int)));
    
    // Launch params buffer
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_launch_params_),
                         sizeof(LaunchParams)));
}

void LidarSimulator::freeBuffers() {
    if (d_vertices_) { cudaFree(reinterpret_cast<void*>(d_vertices_)); d_vertices_ = 0; }
    if (d_triangles_) { cudaFree(reinterpret_cast<void*>(d_triangles_)); d_triangles_ = 0; }
    if (d_lidar_positions_) { cudaFree(reinterpret_cast<void*>(d_lidar_positions_)); d_lidar_positions_ = 0; }
    if (d_lidar_orientations_) { cudaFree(reinterpret_cast<void*>(d_lidar_orientations_)); d_lidar_orientations_ = 0; }
    if (d_hit_points_) { cudaFree(reinterpret_cast<void*>(d_hit_points_)); d_hit_points_ = 0; }
    if (d_hit_distances_) { cudaFree(reinterpret_cast<void*>(d_hit_distances_)); d_hit_distances_ = 0; }
    if (d_hit_face_ids_) { cudaFree(reinterpret_cast<void*>(d_hit_face_ids_)); d_hit_face_ids_ = 0; }
    if (d_launch_params_) { cudaFree(reinterpret_cast<void*>(d_launch_params_)); d_launch_params_ = 0; }
    if (d_gas_output_buffer_) { cudaFree(reinterpret_cast<void*>(d_gas_output_buffer_)); d_gas_output_buffer_ = 0; }
    if (d_ias_output_buffer_) { cudaFree(reinterpret_cast<void*>(d_ias_output_buffer_)); d_ias_output_buffer_ = 0; }
}

void LidarSimulator::buildAccelerationStructure() {
    // For batch processing, we need to create triangle indices for each batch
    // Each batch has the same topology but different vertex offsets
    
    // Build options
    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_UPDATE |  // Allow updates
                               OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;
    
    // Create expanded triangle indices for all batches
    // Each batch needs its own set of triangles with vertex offset
    int total_triangles = batch_size_ * num_triangles_;
    std::vector<int3> expanded_triangles(total_triangles);
    
    // Copy triangles from device to host first
    std::vector<int3> base_triangles(num_triangles_);
    CUDA_CHECK(cudaMemcpy(
        base_triangles.data(),
        reinterpret_cast<void*>(d_triangles_),
        num_triangles_ * sizeof(int3),
        cudaMemcpyDeviceToHost
    ));
    
    // Create triangles for each batch with proper vertex offsets
    for (int b = 0; b < batch_size_; ++b) {
        int vertex_offset = b * num_vertices_;
        for (int t = 0; t < num_triangles_; ++t) {
            int expanded_idx = b * num_triangles_ + t;
            expanded_triangles[expanded_idx].x = base_triangles[t].x + vertex_offset;
            expanded_triangles[expanded_idx].y = base_triangles[t].y + vertex_offset;
            expanded_triangles[expanded_idx].z = base_triangles[t].z + vertex_offset;
        }
    }
    
    // Allocate device memory for expanded triangles
    CUdeviceptr d_expanded_triangles;
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_expanded_triangles),
        total_triangles * sizeof(int3)
    ));
    
    // Copy expanded triangles to device
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_expanded_triangles),
        expanded_triangles.data(),
        total_triangles * sizeof(int3),
        cudaMemcpyHostToDevice
    ));
    
    // Build input for triangle mesh
    OptixBuildInput triangle_input = {};
    triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    
    // Vertex buffer - all batches concatenated
    CUdeviceptr d_vertices = d_vertices_;
    triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangle_input.triangleArray.vertexBuffers = &d_vertices;
    triangle_input.triangleArray.numVertices = batch_size_ * num_vertices_;
    triangle_input.triangleArray.vertexStrideInBytes = sizeof(float3);
    
    // Index buffer - expanded for all batches
    triangle_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    triangle_input.triangleArray.indexBuffer = d_expanded_triangles;
    triangle_input.triangleArray.numIndexTriplets = total_triangles;
    triangle_input.triangleArray.indexStrideInBytes = sizeof(int3);
    
    uint32_t triangle_input_flags[1] = {OPTIX_GEOMETRY_FLAG_NONE};
    triangle_input.triangleArray.flags = triangle_input_flags;
    triangle_input.triangleArray.numSbtRecords = 1;
    
    // Query memory requirements
    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
        context_,
        &accel_options,
        &triangle_input,
        1,
        &gas_buffer_sizes
    ));
    
    // Allocate temp buffer
    CUdeviceptr d_temp_buffer;
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_temp_buffer),
        gas_buffer_sizes.tempSizeInBytes
    ));
    
    // Allocate output buffer
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_gas_output_buffer_),
        gas_buffer_sizes.outputSizeInBytes
    ));
    
    // Build acceleration structure
    OptixTraversableHandle gas_handle;
    OPTIX_CHECK(optixAccelBuild(
        context_,
        stream_,
        &accel_options,
        &triangle_input,
        1,
        d_temp_buffer,
        gas_buffer_sizes.tempSizeInBytes,
        d_gas_output_buffer_,
        gas_buffer_sizes.outputSizeInBytes,
        &gas_handle,
        nullptr,
        0
    ));
    
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    
    // Free temp buffers
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_temp_buffer)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_expanded_triangles)));
    
    // Store handle (for now, single GAS for all)
    gas_handles_.resize(1);
    gas_handles_[0] = gas_handle;
    ias_handle_ = gas_handle;  // Use GAS directly for now
    
    // Verbose output disabled for production use
    // std::cout << "Acceleration structure built successfully" << std::endl;
}

void LidarSimulator::raycastBatch(
    const float* lidar_positions,
    const float* lidar_orientations,
    float* output_points,
    float* output_distances,
    int* output_face_ids,
    bool input_on_device,
    bool output_on_device) {
    std::lock_guard<std::mutex> lock(raycast_mutex_);

    cudaMemcpyKind input_copy_kind = input_on_device ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;
    cudaMemcpyKind output_copy_kind = output_on_device ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost;

    // Copy LiDAR positions to device
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_lidar_positions_),
        lidar_positions,
        batch_size_ * sizeof(float3),
        input_copy_kind
    ));

    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_lidar_orientations_),
        lidar_orientations,
        batch_size_ * sizeof(float3),
        input_copy_kind
    ));
    
    // Setup launch parameters
    LaunchParams params;
    params.hit_points = reinterpret_cast<float3*>(d_hit_points_);
    params.hit_distances = reinterpret_cast<float*>(d_hit_distances_);
    params.hit_face_ids = reinterpret_cast<int*>(d_hit_face_ids_);
    params.vertices = reinterpret_cast<float3*>(d_vertices_);
    params.triangles = reinterpret_cast<int3*>(d_triangles_);
    params.num_vertices = num_vertices_;
    params.num_triangles = num_triangles_;
    params.lidar_positions = reinterpret_cast<float3*>(d_lidar_positions_);
    params.lidar_orientations = reinterpret_cast<float3*>(d_lidar_orientations_);
    params.num_rays = num_rays_;
    params.num_vertical_beams = num_vertical_beams_;
    params.num_horizontal_samples = num_horizontal_samples_;
    params.vertical_fov_min = vertical_fov_min_;
    params.vertical_fov_max = vertical_fov_max_;
    params.horizontal_fov_min = horizontal_fov_min_;
    params.horizontal_fov_max = horizontal_fov_max_;
    params.max_range = max_range_;
    params.batch_size = batch_size_;
    params.traversable_handle = ias_handle_;
    
    // Copy params to device
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_launch_params_),
        &params,
        sizeof(LaunchParams),
        cudaMemcpyHostToDevice
    ));
    
    // Launch OptiX ray tracing
    OPTIX_CHECK(optixLaunch(
        pipeline_,
        stream_,
        d_launch_params_,
        sizeof(LaunchParams),
        &sbt_,
        num_rays_,
        batch_size_,
        1
    ));
    
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    
    // Copy results back to host
    size_t points_size = batch_size_ * num_rays_ * sizeof(float3);
    size_t distances_size = batch_size_ * num_rays_ * sizeof(float);
    size_t face_ids_size = batch_size_ * num_rays_ * sizeof(int);
    
    CUDA_CHECK(cudaMemcpy(
        output_points,
        reinterpret_cast<void*>(d_hit_points_),
        points_size,
        output_copy_kind
    ));
    
    CUDA_CHECK(cudaMemcpy(
        output_distances,
        reinterpret_cast<void*>(d_hit_distances_),
        distances_size,
        output_copy_kind
    ));
    
    CUDA_CHECK(cudaMemcpy(
        output_face_ids,
        reinterpret_cast<void*>(d_hit_face_ids_),
        face_ids_size,
        output_copy_kind
    ));
}

void LidarSimulator::cleanup() {
    freeBuffers();
    
    // Free SBT records - these were allocated in createSBT
    // Note: cudaFree with 0 is a no-op, so safe to call
    if (sbt_.raygenRecord) {
        cudaFree(reinterpret_cast<void*>(sbt_.raygenRecord));
        sbt_.raygenRecord = 0;
    }
    if (sbt_.missRecordBase) {
        cudaFree(reinterpret_cast<void*>(sbt_.missRecordBase));
        sbt_.missRecordBase = 0;
    }
    if (sbt_.hitgroupRecordBase) {
        cudaFree(reinterpret_cast<void*>(sbt_.hitgroupRecordBase));
        sbt_.hitgroupRecordBase = 0;
    }
    
    // Destroy program groups
    if (raygen_pg_) {
        optixProgramGroupDestroy(raygen_pg_);
        raygen_pg_ = nullptr;
    }
    if (miss_pg_) {
        optixProgramGroupDestroy(miss_pg_);
        miss_pg_ = nullptr;
    }
    if (hitgroup_pg_) {
        optixProgramGroupDestroy(hitgroup_pg_);
        hitgroup_pg_ = nullptr;
    }
    if (anyhit_pg_) {
        anyhit_pg_ = nullptr;
    }
    
    // Destroy pipeline and module
    if (pipeline_) {
        optixPipelineDestroy(pipeline_);
        pipeline_ = nullptr;
    }
    if (module_) {
        optixModuleDestroy(module_);
        module_ = nullptr;
    }
    
    // Destroy context
    if (context_) {
        optixDeviceContextDestroy(context_);
        context_ = nullptr;
    }
    
    // Destroy CUDA stream
    if (stream_) {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
}
