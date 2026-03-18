#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cstdint>
#include <string>
#include "LidarSimulator.h"

namespace py = pybind11;

namespace {

bool is_torch_tensor(const py::handle& obj) {
    return py::hasattr(obj, "data_ptr") &&
           py::hasattr(obj, "is_cuda") &&
           py::hasattr(obj, "dtype") &&
           py::hasattr(obj, "shape") &&
           py::hasattr(obj, "is_contiguous");
}

std::string dtype_string(const py::handle& obj) {
    return py::str(obj.attr("dtype"));
}

void validate_contiguous_tensor(const py::handle& tensor, const char* name) {
    if (!tensor.attr("is_contiguous")().cast<bool>()) {
        throw std::runtime_error(std::string(name) + " must be contiguous");
    }
}

}  // namespace

class LidarSimulatorWrapper {
public:
    LidarSimulatorWrapper() {
        simulator_ = new LidarSimulator();
    }
    
    ~LidarSimulatorWrapper() {
        delete simulator_;
    }
    
    void initialize(int device_id = -1) {
        simulator_->initialize(device_id);
    }
    
    void setLidarConfig(
        int num_vertical_beams,
        int num_horizontal_samples,
        float vertical_fov_min,
        float vertical_fov_max,
        float horizontal_fov_min,
        float horizontal_fov_max,
        float max_range) {
        
        simulator_->setLidarConfig(
            num_vertical_beams,
            num_horizontal_samples,
            vertical_fov_min,
            vertical_fov_max,
            horizontal_fov_min,
            horizontal_fov_max,
            max_range
        );
    }
    
    void setMeshBatch(
        py::object vertices,
        py::object triangles,
        int batch_size = -1) {

        const bool vertices_is_torch = is_torch_tensor(vertices);
        const bool triangles_is_torch = is_torch_tensor(triangles);
        if (vertices_is_torch != triangles_is_torch) {
            throw std::runtime_error("vertices and triangles must both be torch tensors or both be numpy arrays");
        }

        if (vertices_is_torch) {
            validate_contiguous_tensor(vertices, "vertices");
            validate_contiguous_tensor(triangles, "triangles");

            const std::string vertices_dtype = dtype_string(vertices);
            const std::string triangles_dtype = dtype_string(triangles);
            if (vertices_dtype != "torch.float32") {
                throw std::runtime_error("vertices dtype must be torch.float32");
            }
            if (triangles_dtype != "torch.int32") {
                throw std::runtime_error("triangles dtype must be torch.int32");
            }

            py::tuple vertices_shape = vertices.attr("shape").cast<py::tuple>();
            py::tuple triangles_shape = triangles.attr("shape").cast<py::tuple>();

            if (vertices_shape.size() != 3 || vertices_shape[2].cast<int>() != 3) {
                throw std::runtime_error("vertices must have shape (batch_size, num_vertices, 3)");
            }
            if (triangles_shape.size() != 2 || triangles_shape[1].cast<int>() != 3) {
                throw std::runtime_error("triangles must have shape (num_triangles, 3)");
            }

            int inferred_batch_size = vertices_shape[0].cast<int>();
            int num_vertices = vertices_shape[1].cast<int>();
            int num_triangles = triangles_shape[0].cast<int>();
            if (batch_size < 0) {
                batch_size = inferred_batch_size;
            }
            if (batch_size != inferred_batch_size) {
                throw std::runtime_error("batch_size does not match vertices first dimension");
            }

            bool vertices_on_device = vertices.attr("is_cuda").cast<bool>();
            bool triangles_on_device = triangles.attr("is_cuda").cast<bool>();
            auto vertices_ptr = reinterpret_cast<const float*>(
                vertices.attr("data_ptr")().cast<std::uintptr_t>()
            );
            auto triangles_ptr = reinterpret_cast<const int*>(
                triangles.attr("data_ptr")().cast<std::uintptr_t>()
            );

            simulator_->setMeshBatch(
                vertices_ptr,
                triangles_ptr,
                batch_size,
                num_vertices,
                num_triangles,
                vertices_on_device,
                triangles_on_device
            );
            return;
        }

        py::array_t<float, py::array::c_style | py::array::forcecast> vertices_arr =
            vertices.cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
        py::array_t<int, py::array::c_style | py::array::forcecast> triangles_arr =
            triangles.cast<py::array_t<int, py::array::c_style | py::array::forcecast>>();

        py::buffer_info vertices_buf = vertices_arr.request();
        py::buffer_info triangles_buf = triangles_arr.request();

        if (vertices_buf.ndim != 3 || vertices_buf.shape[2] != 3) {
            throw std::runtime_error("Vertices must be 3D array (batch_size, num_vertices, 3)");
        }
        if (triangles_buf.ndim != 2 || triangles_buf.shape[1] != 3) {
            throw std::runtime_error("Triangles must be 2D array (num_triangles, 3)");
        }

        int inferred_batch_size = static_cast<int>(vertices_buf.shape[0]);
        int num_vertices = static_cast<int>(vertices_buf.shape[1]);
        int num_triangles = static_cast<int>(triangles_buf.shape[0]);
        if (batch_size < 0) {
            batch_size = inferred_batch_size;
        }
        if (batch_size != inferred_batch_size) {
            throw std::runtime_error("batch_size does not match vertices first dimension");
        }

        simulator_->setMeshBatch(
            static_cast<float*>(vertices_buf.ptr),
            static_cast<int*>(triangles_buf.ptr),
            batch_size,
            num_vertices,
            num_triangles,
            false,
            false
        );
    }

    py::tuple raycastBatch(
        py::object lidar_positions,
        py::object lidar_orientations) {

        const bool positions_is_torch = is_torch_tensor(lidar_positions);
        const bool orientations_is_torch = is_torch_tensor(lidar_orientations);
        if (positions_is_torch != orientations_is_torch) {
            throw std::runtime_error("lidar_positions and lidar_orientations must both be torch tensors or both be numpy arrays");
        }

        int num_rays = simulator_->getNumRays();

        if (positions_is_torch) {
            validate_contiguous_tensor(lidar_positions, "lidar_positions");
            validate_contiguous_tensor(lidar_orientations, "lidar_orientations");

            if (dtype_string(lidar_positions) != "torch.float32") {
                throw std::runtime_error("lidar_positions dtype must be torch.float32");
            }
            if (dtype_string(lidar_orientations) != "torch.float32") {
                throw std::runtime_error("lidar_orientations dtype must be torch.float32");
            }

            py::tuple pos_shape = lidar_positions.attr("shape").cast<py::tuple>();
            py::tuple ori_shape = lidar_orientations.attr("shape").cast<py::tuple>();
            if (pos_shape.size() != 2 || pos_shape[1].cast<int>() != 3) {
                throw std::runtime_error("lidar_positions must have shape (batch_size, 3)");
            }
            if (ori_shape.size() != 2 || ori_shape[1].cast<int>() != 3) {
                throw std::runtime_error("lidar_orientations must have shape (batch_size, 3)");
            }

            const int batch_size = pos_shape[0].cast<int>();
            if (ori_shape[0].cast<int>() != batch_size) {
                throw std::runtime_error("lidar_positions and lidar_orientations batch size must match");
            }

            const bool positions_on_device = lidar_positions.attr("is_cuda").cast<bool>();
            const bool orientations_on_device = lidar_orientations.attr("is_cuda").cast<bool>();
            if (positions_on_device != orientations_on_device) {
                throw std::runtime_error("lidar_positions and lidar_orientations must be on same device type");
            }

            py::module_ torch = py::module_::import("torch");
            py::object device = lidar_positions.attr("device");

            py::object hit_points = torch.attr("empty")(
                py::make_tuple(batch_size, num_rays, 3),
                py::arg("dtype") = torch.attr("float32"),
                py::arg("device") = device
            );
            py::object hit_distances = torch.attr("empty")(
                py::make_tuple(batch_size, num_rays),
                py::arg("dtype") = torch.attr("float32"),
                py::arg("device") = device
            );
            py::object hit_face_ids = torch.attr("empty")(
                py::make_tuple(batch_size, num_rays),
                py::arg("dtype") = torch.attr("int32"),
                py::arg("device") = device
            );

            simulator_->raycastBatch(
                reinterpret_cast<const float*>(lidar_positions.attr("data_ptr")().cast<std::uintptr_t>()),
                reinterpret_cast<const float*>(lidar_orientations.attr("data_ptr")().cast<std::uintptr_t>()),
                reinterpret_cast<float*>(hit_points.attr("data_ptr")().cast<std::uintptr_t>()),
                reinterpret_cast<float*>(hit_distances.attr("data_ptr")().cast<std::uintptr_t>()),
                reinterpret_cast<int*>(hit_face_ids.attr("data_ptr")().cast<std::uintptr_t>()),
                positions_on_device,
                positions_on_device
            );

            return py::make_tuple(hit_points, hit_distances, hit_face_ids);
        }

        py::array_t<float, py::array::c_style | py::array::forcecast> positions_arr =
            lidar_positions.cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
        py::array_t<float, py::array::c_style | py::array::forcecast> orientations_arr =
            lidar_orientations.cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();

        py::buffer_info pos_buf = positions_arr.request();
        py::buffer_info ori_buf = orientations_arr.request();

        if (pos_buf.ndim != 2 || pos_buf.shape[1] != 3) {
            throw std::runtime_error("LiDAR positions must be (batch_size, 3)");
        }
        if (ori_buf.ndim != 2 || ori_buf.shape[1] != 3) {
            throw std::runtime_error("LiDAR orientations must be (batch_size, 3)");
        }

        int batch_size = static_cast<int>(pos_buf.shape[0]);
        if (ori_buf.shape[0] != pos_buf.shape[0]) {
            throw std::runtime_error("LiDAR positions/orientations batch size mismatch");
        }

        auto hit_points = py::array_t<float>({batch_size, num_rays, 3});
        auto hit_distances = py::array_t<float>({batch_size, num_rays});
        auto hit_face_ids = py::array_t<int>({batch_size, num_rays});

        py::buffer_info points_buf = hit_points.request();
        py::buffer_info distances_buf = hit_distances.request();
        py::buffer_info face_ids_buf = hit_face_ids.request();

        simulator_->raycastBatch(
            static_cast<float*>(pos_buf.ptr),
            static_cast<float*>(ori_buf.ptr),
            static_cast<float*>(points_buf.ptr),
            static_cast<float*>(distances_buf.ptr),
            static_cast<int*>(face_ids_buf.ptr),
            false,
            false
        );

        return py::make_tuple(hit_points, hit_distances, hit_face_ids);
    }
    
    void cleanup() {
        simulator_->cleanup();
    }
    
    int getNumRays() const {
        return simulator_->getNumRays();
    }
    
private:
    LidarSimulator* simulator_;
};

PYBIND11_MODULE(_optix_lidar_raycast, m) {
    m.doc() = "OptiX-based LiDAR raycasting simulator for batch mesh inference";
    
    py::class_<LidarSimulatorWrapper>(m, "LidarSimulator")
        .def(py::init<>())
        .def("initialize", &LidarSimulatorWrapper::initialize,
             py::arg("device_id") = -1,
             "Initialize OptiX context and pipeline\n"
             "Args:\n"
             "    device_id: CUDA device ID (-1 = use current PyTorch device)")
        .def("set_lidar_config", &LidarSimulatorWrapper::setLidarConfig,
             py::arg("num_vertical_beams"),
             py::arg("num_horizontal_samples"),
             py::arg("vertical_fov_min"),
             py::arg("vertical_fov_max"),
             py::arg("horizontal_fov_min"),
             py::arg("horizontal_fov_max"),
             py::arg("max_range"),
             "Configure LiDAR scanner parameters (FOV in degrees)")
        .def("set_mesh_batch", &LidarSimulatorWrapper::setMeshBatch,
             py::arg("vertices"),
             py::arg("triangles"),
               py::arg("batch_size") = -1,
             "Set mesh data for batch processing\n"
             "Args:\n"
               "    vertices: torch.float32 tensor or float32 numpy array, shape (batch_size, num_vertices, 3)\n"
               "    triangles: torch.int32 tensor or int32 numpy array, shape (num_triangles, 3)\n"
               "    batch_size: optional override, default infer from vertices")
        .def("raycast_batch", &LidarSimulatorWrapper::raycastBatch,
             py::arg("lidar_positions"),
             py::arg("lidar_orientations"),
             "Perform raycasting for all meshes in batch\n"
             "Args:\n"
             "    lidar_positions: (batch_size, 3) sensor positions\n"
             "    lidar_orientations: (batch_size, 3) Euler angles in degrees\n"
             "Returns:\n"
             "    tuple: (hit_points, hit_distances, hit_face_ids)\n"
             "        hit_points: (batch_size, num_rays, 3)\n"
             "        hit_distances: (batch_size, num_rays)\n"
             "        hit_face_ids: (batch_size, num_rays)")
        .def("cleanup", &LidarSimulatorWrapper::cleanup,
             "Release all resources")
        .def("get_num_rays", &LidarSimulatorWrapper::getNumRays,
             "Get total number of rays per scan");
}
