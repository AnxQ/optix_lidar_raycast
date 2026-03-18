"""
OptiX LiDAR Raycasting Simulator

This module provides a high-performance LiDAR simulation using NVIDIA OptiX ray tracing.
It supports batch processing of multiple meshes with a single LiDAR configuration.
"""

import numpy as np
import torch
from typing import Tuple, Optional
try:
    from . import _optix_lidar_raycast as _C
except ImportError:
    import _optix_lidar_raycast as _C

class OptiXLidarSimulator:
    """
    High-performance LiDAR simulator using OptiX ray tracing.
    
    Features:
    - Batch processing: simulate multiple meshes in parallel
    - GPU acceleration: OptiX hardware ray tracing
    - PyTorch integration: zero-copy tensor interface
    
    Example:
        >>> simulator = OptiXLidarSimulator(
        ...     num_vertical_beams=32,
        ...     num_horizontal_samples=1024,
        ...     vertical_fov=(-16, 16),
        ...     horizontal_fov=(0, 360),
        ...     max_range=100.0
        ... )
        >>> 
        >>> # Set mesh data (batch_size=16, SMPL mesh)
        >>> vertices = torch.randn(16, 6890, 3)  # (B, V, 3)
        >>> faces = torch.randint(0, 6890, (13776, 3))  # (F, 3)
        >>> simulator.set_mesh_batch(vertices, faces)
        >>> 
        >>> # Define LiDAR poses
        >>> lidar_positions = torch.randn(16, 3)  # (B, 3)
        >>> lidar_orientations = torch.zeros(16, 3)  # (B, 3) Euler angles
        >>> 
        >>> # Simulate point clouds
        >>> points, distances, face_ids = simulator.simulate(
        ...     lidar_positions, lidar_orientations
        ... )
        >>> 
        >>> print(points.shape)  # (16, 32768, 3)
    """
    
    def __init__(
        self,
        num_vertical_beams: int = 32,
        num_horizontal_samples: int = 1024,
        vertical_fov: Tuple[float, float] = (-16, 16),
        horizontal_fov: Tuple[float, float] = (0, 360),
        max_range: float = 100.0
    ):
        """
        Initialize OptiX LiDAR simulator.
        
        Args:
            num_vertical_beams: Number of vertical scan lines (elevation resolution)
            num_horizontal_samples: Number of horizontal samples per scan line (azimuth resolution)
            vertical_fov: Vertical field of view in degrees (min, max)
            horizontal_fov: Horizontal field of view in degrees (min, max)
            max_range: Maximum detection range in meters
        """
        self.simulator = _C.LidarSimulator()
        
        # Store configuration for lazy initialization
        # DO NOT initialize CUDA/OptiX here to avoid issues with forked subprocesses
        # Initialization will happen on first use (set_mesh_batch or simulate)
        self.num_vertical_beams = num_vertical_beams
        self.num_horizontal_samples = num_horizontal_samples
        self.num_rays = num_vertical_beams * num_horizontal_samples
        self.vertical_fov = vertical_fov
        self.horizontal_fov = horizontal_fov
        self.max_range = max_range
        
        self.batch_size = None
        self._initialized = False
        
    def _ensure_initialized(self):
        """Lazy initialization to be called before first use.
        
        This avoids CUDA initialization in __init__, which would fail
        in DataLoader worker processes (forked from main process).
        """
        if not self._initialized:
            # Always use device_id = -1 to let CUDA runtime detect current device
            # DO NOT call any torch.cuda functions here as they will fail in forked processes
            device_id = -1
            
            try:
                self.simulator.initialize(device_id)
            except RuntimeError as e:
                error_msg = str(e)
                if "forked" in error_msg.lower() or "CUDA" in error_msg:
                    raise RuntimeError(
                        f"Failed to initialize OptiX in worker process. "
                        f"This is a known limitation when using DataLoader with num_workers > 0 in fork mode.\n"
                        f"Solutions:\n"
                        f"  1. Set num_workers=0 in DataLoader (single process)\n"
                        f"  2. Use multiprocessing_context='spawn' in DataLoader\n"
                        f"  3. Set persistent_workers=True in DataLoader (may help)\n"
                        f"Original error: {error_msg}"
                    ) from e
                else:
                    raise
            
            self.simulator.set_lidar_config(
                self.num_vertical_beams,
                self.num_horizontal_samples,
                self.vertical_fov[0],
                self.vertical_fov[1],
                self.horizontal_fov[0],
                self.horizontal_fov[1],
                self.max_range
            )
            
            self._initialized = True
            # Verbose output disabled for production use
            # print(f"OptiXLidarSimulator initialized: {self.num_vertical_beams}x{self.num_horizontal_samples} = {self.num_rays} rays")
    
    def set_mesh_batch(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor
    ):
        """
        Set mesh data for batch processing.
        
        Args:
            vertices: Vertex positions (batch_size, num_vertices, 3) on CPU or GPU
            faces: Triangle indices (num_triangles, 3) on CPU or GPU
        """
        # Ensure simulator is initialized (lazy initialization)
        self._ensure_initialized()
        
        if not isinstance(vertices, torch.Tensor):
            vertices = torch.tensor(vertices, dtype=torch.float32)
        if not isinstance(faces, torch.Tensor):
            faces = torch.tensor(faces, dtype=torch.int32)
        
        # Ensure correct dtype and contiguous memory.
        vertices = vertices.to(dtype=torch.float32).contiguous()
        faces = faces.to(dtype=torch.int32).contiguous()

        # Validate shapes
        assert vertices.ndim == 3, f"Vertices must be 3D (B, V, 3), got {tuple(vertices.shape)}"
        assert faces.ndim == 2, f"Faces must be 2D (F, 3), got {tuple(faces.shape)}"
        assert vertices.shape[2] == 3, "Vertices must have 3 coordinates"
        assert faces.shape[1] == 3, "Faces must have 3 indices"

        self.batch_size = int(vertices.shape[0])
        self.num_vertices = int(vertices.shape[1])
        self.num_triangles = int(faces.shape[0])

        # Directly pass torch tensors to backend (CPU/CUDA supported).
        self.simulator.set_mesh_batch(vertices, faces, self.batch_size)
    
    def simulate(
        self,
        lidar_positions: torch.Tensor,
        lidar_orientations: torch.Tensor,
        return_torch: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Simulate LiDAR point clouds for all meshes in batch.
        
        Args:
            lidar_positions: Sensor positions (batch_size, 3)
            lidar_orientations: Sensor orientations as Euler angles in degrees (batch_size, 3)
                               [roll, pitch, yaw] in ZYX convention
            return_torch: If True, return PyTorch tensors; otherwise return NumPy arrays
        
        Returns:
            hit_points: Hit positions (batch_size, num_rays, 3)
            hit_distances: Ray distances (batch_size, num_rays)
            hit_face_ids: Face indices (batch_size, num_rays), -1 for misses
        """
        # Ensure simulator is initialized (lazy initialization)
        self._ensure_initialized()
        
        if self.batch_size is None:
            raise RuntimeError("Call set_mesh_batch() first")
        
        if not isinstance(lidar_positions, torch.Tensor):
            lidar_positions = torch.tensor(lidar_positions, dtype=torch.float32)
        if not isinstance(lidar_orientations, torch.Tensor):
            lidar_orientations = torch.tensor(lidar_orientations, dtype=torch.float32)
        
        lidar_positions = lidar_positions.to(dtype=torch.float32).contiguous()
        lidar_orientations = lidar_orientations.to(dtype=torch.float32).contiguous()

        # Validate shapes
        assert tuple(lidar_positions.shape) == (self.batch_size, 3), \
            f"Positions shape mismatch: expected ({self.batch_size}, 3), got {tuple(lidar_positions.shape)}"
        assert tuple(lidar_orientations.shape) == (self.batch_size, 3), \
            f"Orientations shape mismatch: expected ({self.batch_size}, 3), got {tuple(lidar_orientations.shape)}"

        # Call C++ raycast. Backend returns torch tensors for torch inputs.
        hit_points, hit_distances, hit_face_ids = self.simulator.raycast_batch(
            lidar_positions,
            lidar_orientations
        )

        if return_torch and isinstance(hit_points, np.ndarray):
            hit_points = torch.from_numpy(hit_points)
            hit_distances = torch.from_numpy(hit_distances)
            hit_face_ids = torch.from_numpy(hit_face_ids)

        if not return_torch and isinstance(hit_points, torch.Tensor):
            hit_points = hit_points.detach().cpu().numpy()
            hit_distances = hit_distances.detach().cpu().numpy()
            hit_face_ids = hit_face_ids.detach().cpu().numpy()
        
        return hit_points, hit_distances, hit_face_ids
    
    def filter_valid_points(
        self,
        hit_points: torch.Tensor,
        hit_face_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Filter out missed rays (face_id == -1).
        
        Args:
            hit_points: Hit positions (batch_size, num_rays, 3)
            hit_face_ids: Face indices (batch_size, num_rays)
        
        Returns:
            List of valid point clouds for each batch item
        """
        valid_points = []
        for i in range(self.batch_size):
            mask = hit_face_ids[i] >= 0
            valid_points.append(hit_points[i][mask])
        return valid_points
    
    def cleanup(self):
        """Release all OptiX resources."""
        self.simulator.cleanup()
    
    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.cleanup()
        except:
            pass


def create_lidar_config(
    lidar_type: str = 'velodyne_hdl32'
) -> dict:
    """
    Create LiDAR configuration presets.
    
    Args:
        lidar_type: One of ['velodyne_hdl32', 'velodyne_hdl64', 'ouster_os1_64', 'custom']
    
    Returns:
        Configuration dictionary for OptiXLidarSimulator
    """
    configs = {
        'velodyne_hdl32': {
            'num_vertical_beams': 32,
            'num_horizontal_samples': 1024,
            'vertical_fov': (-30.67, 10.67),
            'horizontal_fov': (0, 360),
            'max_range': 100.0,
        },
        'velodyne_hdl64': {
            'num_vertical_beams': 64,
            'num_horizontal_samples': 2048,
            'vertical_fov': (-24.9, 2.0),
            'horizontal_fov': (0, 360),
            'max_range': 120.0,
        },
        'ouster_os1_64': {
            'num_vertical_beams': 64,
            'num_horizontal_samples': 1024,
            'vertical_fov': (-22.5, 22.5),
            'horizontal_fov': (0, 360),
            'max_range': 120.0,
        },
        'dense': {
            'num_vertical_beams': 60,
            'num_horizontal_samples': 2650,
            'vertical_fov': (-30, 30),
            'horizontal_fov': (0, 360),
            'max_range': 100.0,
        },
    }
    
    if lidar_type not in configs:
        raise ValueError(f"Unknown LiDAR type: {lidar_type}. "
                        f"Available: {list(configs.keys())}")
    
    return configs[lidar_type]
