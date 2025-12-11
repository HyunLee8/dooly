"""
Drone Control System
Combines SLAM mapping with IMU-SLAM sensor fusion for robust drone navigation
"""

# SLAM Module - import from actual files
from .slam.orbslam3_wrapper import (
    ORBSLAM3VIO,
    IMUMeasurement,
    Pose,
    CameraIntrinsics,
    TrackingStatus
)

from .slam.slam_integration import (
    SLAMIntegration,
    SLAMConfig
)

# State Estimation Module
from .state_estimation import (
    IMUSLAMFusion,
    IMUData,
    SLAMPose,
    FusedPose
)

# Tello Control Module
from .tello import (
    TelloController,
    TelloInterface,
    FlightMode,
    TelloState
)

__version__ = "1.0.0"
__author__ = "Drone Control Team"

__all__ = [
    # SLAM exports
    "ORBSLAM3VIO",
    "IMUMeasurement",
    "Pose",
    "CameraIntrinsics",
    "TrackingStatus",
    "SLAMIntegration",
    "SLAMConfig",
    
    # State Estimation exports
    "IMUSLAMFusion",
    "IMUData",
    "SLAMPose",
    "FusedPose",
    
    # Tello Control exports
    "TelloController",
    "TelloInterface",
    "FlightMode",
    "TelloState"
]


# Convenience function to convert SLAM Pose to ESKF SLAMPose
def slam_pose_to_eskf(slam_pose: Pose, tracking_status: TrackingStatus) -> SLAMPose:
    """
    Convert SLAM Pose to ESKF SLAMPose format
    
    Args:
        slam_pose: Pose from ORB-SLAM3
        tracking_status: SLAM tracking status
        
    Returns:
        SLAMPose for ESKF update
    """
    return SLAMPose(
        x=slam_pose.position[0],
        y=slam_pose.position[1],
        z=slam_pose.position[2],
        qw=slam_pose.quaternion[0],
        qx=slam_pose.quaternion[1],
        qy=slam_pose.quaternion[2],
        qz=slam_pose.quaternion[3],
        tracking_status=(tracking_status == TrackingStatus.OK),
        timestamp=slam_pose.timestamp
    )


# Convenience function to convert raw IMU to ESKF IMUData
def raw_imu_to_eskf(ax: float, ay: float, az: float, 
                     gx: float, gy: float, gz: float, 
                     timestamp: float) -> IMUData:
    """
    Convert raw IMU measurements to ESKF IMUData format
    
    Args:
        ax, ay, az: Accelerometer readings (m/s^2)
        gx, gy, gz: Gyroscope readings (rad/s)
        timestamp: Measurement timestamp
        
    Returns:
        IMUData for ESKF prediction
    """
    return IMUData(
        ax=ax, ay=ay, az=az,
        gx=gx, gy=gy, gz=gz,
        timestamp=timestamp
    )