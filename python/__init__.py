"""
OptiX LiDAR Raycasting Simulator
High-performance LiDAR simulation using NVIDIA OptiX
"""

__version__ = '0.1.0'

from .lidar_raycast import OptiXLidarSimulator, create_lidar_config

__all__ = [
    'OptiXLidarSimulator',
    'create_lidar_config'
]
