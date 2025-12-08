"""
SLAM Module for Drone Navigation
Visual-Inertial SLAM using ORB-SLAM3
"""

from .orbslam3_wrapper import (
    ORBSLAM3VIO,
    IMUMeasurement,
    Pose,
    CameraIntrinsics,
    TrackingStatus
)

from .slam_integration import (
    SLAMIntegration,
    SLAMConfig
)

__version__ = "1.0.0"
__all__ = [
    "ORBSLAM3VIO",
    "IMUMeasurement", 
    "Pose",
    "CameraIntrinsics",
    "TrackingStatus",
    "SLAMIntegration",
    "SLAMConfig"
]