# optix_lidar_raycast

OptiX-based LiDAR raycasting backend for pytorch (NVIDIA GPU required), much faster than the `RayCastingScene` in open3d. Designed for augmentation & synthesis of deformable non-rigid bodies (such as human mesh). 

## Installation
- Ensure the dependencies:
    - CUDA Toolkit >= 11.8
    - Pytorch >= 1.8.0
    - CMake >= 3.18

- Download the [OptiX SDK](https://developer.nvidia.com/designworks/optix/download), and set `OptiX_INSTALL_DIR` before installation (8.1.0 is recommended):

```bash
export OptiX_INSTALL_DIR=/path/to/NVIDIA-OptiX-SDK-8.1.0-linux64-x86_64
cd optix_lidar_raycast
pip install -e . --no-deps
```
## Basic Usage

```python
import torch
from optix_lidar_raycast import OptiXLidarSimulator

# 1) Create simulator
sim = OptiXLidarSimulator(
    num_vertical_beams=16,
    num_horizontal_samples=256,
    vertical_fov=(-20.0, 20.0),
    horizontal_fov=(0.0, 360.0),
    max_range=30.0,
)

# 2) Set a simple plane mesh (batch size = 1)
vertices = torch.tensor(
    [[[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]]],
    dtype=torch.float32,
    device="cuda",
)
faces = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int32, device="cuda")
sim.set_mesh_batch(vertices, faces)

# 3) Simulate one LiDAR scan
positions = torch.tensor([[0.0, 0.0, 1.5]], dtype=torch.float32, device="cuda")
orientations = torch.tensor([[0.0, -90.0, 0.0]], dtype=torch.float32, device="cuda")
points, distances, face_ids = sim.simulate(positions, orientations)

print(points.shape, distances.shape, face_ids.shape)
```

Expected output shape example:
- `points`: `[1, 4096, 3]`
- `distances`: `[1, 4096]`
- `face_ids`: `[1, 4096]`

## Notes

- If you are using Docker, add `-e NVIDIA_DRIVER_CAPABILITIES=all` for the container.
