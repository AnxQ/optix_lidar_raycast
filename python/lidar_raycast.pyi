"""
Type stubs for lidar_raycast module.
This file helps IDEs provide autocompletion and type checking.
"""

from typing import Tuple, Dict, Any, Optional
import torch
import numpy as np
import numpy.typing as npt

class OptiXLidarSimulator:
    """
    High-performance LiDAR simulator using OptiX ray tracing.
    
    Features:
        - Batch processing: simulate multiple meshes in parallel
        - GPU acceleration: OptiX hardware ray tracing
        - PyTorch integration: zero-copy tensor interface
        - Lazy initialization: CUDA/OptiX initialized on first use to support fork mode
    """
    
    num_vertical_beams: int
    num_horizontal_samples: int
    num_rays: int
    batch_size: Optional[int]
    vertical_fov: Tuple[float, float]
    horizontal_fov: Tuple[float, float]
    max_range: float
    _initialized: bool
    
    def __init__(
        self,
        num_vertical_beams: int = 32,
        num_horizontal_samples: int = 1024,
        vertical_fov: Tuple[float, float] = (-16, 16),
        horizontal_fov: Tuple[float, float] = (0, 360),
        max_range: float = 100.0
    ) -> None:
        """
        Initialize OptiX LiDAR simulator.
        
        IMPORTANT: CUDA/OptiX initialization is deferred until first use
        to avoid issues with DataLoader fork mode. The actual initialization
        happens in _ensure_initialized(), called by set_mesh_batch() or simulate().
        
        Args:
            num_vertical_beams: Number of vertical scan lines (elevation resolution)
            num_horizontal_samples: Number of horizontal samples per scan line (azimuth resolution)
            vertical_fov: Vertical field of view in degrees (min, max)
            horizontal_fov: Horizontal field of view in degrees (min, max)
            max_range: Maximum detection range in meters
        """
        ...
    
    def _ensure_initialized(self) -> None:
        """
        Lazy initialization of CUDA/OptiX resources.
        
        This is called automatically by set_mesh_batch() and simulate().
        Defers initialization to avoid fork issues in DataLoader workers.
        """
        ...
    
    def set_mesh_batch(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor
    ) -> None:
        """
        Set mesh data for batch processing.
        
        Args:
            vertices: Vertex positions (batch_size, num_vertices, 3) on CPU or GPU
            faces: Triangle indices (num_triangles, 3) on CPU or GPU
        """
        ...
    
    def simulate(
        self,
        positions: torch.Tensor,
        orientations: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform LiDAR simulation.
        
        Args:
            positions: Sensor positions (batch_size, 3)
            orientations: Sensor orientations as Euler angles in degrees (batch_size, 3)
                         Format: (roll, pitch, yaw)
        
        Returns:
            Tuple of:
                - points: Hit point coordinates (batch_size, num_rays, 3)
                - distances: Distances to hits (batch_size, num_rays)
                - face_ids: Face indices (batch_size, num_rays), -1 for misses
        """
        ...

def create_lidar_config(
    preset: str = 'velodyne_hdl32'
) -> Dict[str, Any]:
    """
    Create LiDAR configuration from preset.
    
    Args:
        preset: Preset name, one of:
            - 'velodyne_hdl32': 32 beams, Velodyne HDL-32E
            - 'velodyne_hdl64': 64 beams, Velodyne HDL-64E
            - 'ouster_os1_64': 64 beams, Ouster OS1-64
            - 'dense': High-resolution dense scan
    
    Returns:
        Configuration dictionary with keys:
            - num_vertical_beams
            - num_horizontal_samples
            - vertical_fov
            - horizontal_fov
            - max_range
    """
    ...
