"""
Type stubs for _optix_lidar_raycast C++ extension module.
This file helps IDEs provide autocompletion and type checking.
"""

from typing import Tuple
import numpy as np
import numpy.typing as npt

class LidarSimulator:
    """OptiX-based LiDAR simulator (C++ backend)."""
    
    def __init__(self) -> None:
        """Initialize the LiDAR simulator."""
        ...
    
    def initialize(self) -> None:
        """Initialize OptiX context and pipeline."""
        ...
    
    def set_lidar_config(
        self,
        num_vertical_beams: int,
        num_horizontal_samples: int,
        vertical_fov_min: float,
        vertical_fov_max: float,
        horizontal_fov_min: float,
        horizontal_fov_max: float,
        max_range: float
    ) -> None:
        """
        Configure LiDAR scanner parameters.
        
        Args:
            num_vertical_beams: Number of vertical scan lines
            num_horizontal_samples: Number of horizontal samples per line
            vertical_fov_min: Minimum vertical FOV in degrees
            vertical_fov_max: Maximum vertical FOV in degrees
            horizontal_fov_min: Minimum horizontal FOV in degrees
            horizontal_fov_max: Maximum horizontal FOV in degrees
            max_range: Maximum detection range in meters
        """
        ...
    
    def set_mesh_batch(
        self,
        vertices: npt.NDArray[np.float32],
        triangles: npt.NDArray[np.int32],
        batch_size: int
    ) -> None:
        """
        Set mesh data for batch processing.
        
        Args:
            vertices: (batch_size, num_vertices, 3) vertex positions
            triangles: (num_triangles, 3) triangle indices
            batch_size: Number of meshes in batch
        """
        ...
    
    def raycast_batch(
        self,
        lidar_positions: npt.NDArray[np.float32],
        lidar_orientations: npt.NDArray[np.float32]
    ) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.int32]]:
        """
        Perform raycasting for all meshes in batch.
        
        Args:
            lidar_positions: (batch_size, 3) sensor positions
            lidar_orientations: (batch_size, 3) Euler angles in degrees (roll, pitch, yaw)
        
        Returns:
            Tuple of:
                - hit_points: (batch_size, num_rays, 3) hit point coordinates
                - hit_distances: (batch_size, num_rays) distances to hits
                - hit_face_ids: (batch_size, num_rays) face indices (-1 for misses)
        """
        ...
    
    def cleanup(self) -> None:
        """Release all OptiX and CUDA resources."""
        ...
    
    def get_num_rays(self) -> int:
        """
        Get total number of rays per scan.
        
        Returns:
            num_vertical_beams * num_horizontal_samples
        """
        ...
