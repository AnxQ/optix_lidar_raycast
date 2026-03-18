"""
Type stubs for optix_lidar_raycast package.
This file helps IDEs provide autocompletion and type checking.
"""

from typing import List

__version__: str

from .lidar_raycast import OptiXLidarSimulator, create_lidar_config

__all__: List[str] = ['OptiXLidarSimulator', 'create_lidar_config']
