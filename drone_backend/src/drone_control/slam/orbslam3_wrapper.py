"""
ORB-SLAM3 VIO Wrapper for Drone SLAM
Python interface to ORB-SLAM3 with monocular-inertial mode
"""
import numpy as np
import cv2
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
from enum import Enum
import threading
import time
from collections import deque

try:
    import orbslam3
except ImportError:
    print("Warning: orbslam3 module not found. Install ORB-SLAM3 Python bindings.")
    orbslam3 = None


class TrackingStatus(Enum):
    """SLAM tracking status"""
    NOT_INITIALIZED = 0
    OK = 1
    LOST = 2


@dataclass
class IMUMeasurement:
    """IMU measurement structure"""
    timestamp: float
    accel: np.ndarray  # [ax, ay, az] m/s^2
    gyro: np.ndarray   # [gx, gy, gz] rad/s
    
    def __post_init__(self):
        self.accel = np.array(self.accel, dtype=np.float64)
        self.gyro = np.array(self.gyro, dtype=np.float64)


@dataclass
class Pose:
    """6DOF pose"""
    position: np.ndarray  # [x, y, z] meters
    quaternion: np.ndarray  # [qw, qx, qy, qz]
    timestamp: float
    
    def to_matrix(self) -> np.ndarray:
        """Convert to 4x4 matrix"""
        T = np.eye(4)
        T[:3, :3] = self._quat_to_rot(self.quaternion)
        T[:3, 3] = self.position
        return T
    
    @staticmethod
    def _quat_to_rot(q):
        qw, qx, qy, qz = q
        return np.array([
            [1-2*(qy**2+qz**2), 2*(qx*qy-qw*qz), 2*(qx*qz+qw*qy)],
            [2*(qx*qy+qw*qz), 1-2*(qx**2+qz**2), 2*(qy*qz-qw*qx)],
            [2*(qx*qz-qw*qy), 2*(qy*qz+qw*qx), 1-2*(qx**2+qy**2)]
        ])


@dataclass
class CameraIntrinsics:
    """Camera calibration"""
    fx: float
    fy: float
    cx: float
    cy: float
    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0


class ORBSLAM3VIO:
    """ORB-SLAM3 VIO wrapper"""
    
    def __init__(self, vocab_path: str, config_path: str, 
                 camera_intrinsics: CameraIntrinsics, verbose: bool = True):
        self.vocab_path = vocab_path
        self.config_path = config_path
        self.camera_intrinsics = camera_intrinsics
        self.verbose = verbose
        
        self.imu_buffer = deque(maxlen=100)
        self.imu_lock = threading.Lock()
        
        self.current_pose: Optional[Pose] = None
        self.tracking_status = TrackingStatus.NOT_INITIALIZED
        self.last_frame_time = None
        self.frame_count = 0
        
        self._init_slam()
    
    def _init_slam(self):
        if orbslam3:
            self.slam = orbslam3.System(
                self.vocab_path, self.config_path,
                orbslam3.Sensor.MONOCULAR_INERTIAL
            )
        else:
            self.slam = None
            print("[SLAM] Running in simulation mode")
    
    def add_imu_measurement(self, imu: IMUMeasurement):
        with self.imu_lock:
            self.imu_buffer.append(imu)
    
    def process_frame(self, frame: np.ndarray, timestamp: float) -> Tuple[Optional[Pose], TrackingStatus]:
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        with self.imu_lock:
            imu_list = list(self.imu_buffer)
            self.imu_buffer.clear()
        
        if self.slam:
            for imu in imu_list:
                self.slam.ProcessIMU(
                    imu.accel[0], imu.accel[1], imu.accel[2],
                    imu.gyro[0], imu.gyro[1], imu.gyro[2],
                    imu.timestamp
                )
            
            pose_mat = self.slam.TrackMonocular(frame, timestamp)
            
            if pose_mat is not None:
                self.current_pose = Pose(
                    position=pose_mat[:3, 3],
                    quaternion=self._rot_to_quat(pose_mat[:3, :3]),
                    timestamp=timestamp
                )
                self.tracking_status = TrackingStatus.OK
            else:
                self.tracking_status = TrackingStatus.LOST
        else:
            self.current_pose = self._sim_pose(timestamp)
            self.tracking_status = TrackingStatus.OK
        
        self.frame_count += 1
        self.last_frame_time = timestamp
        
        return self.current_pose, self.tracking_status
    
    def _sim_pose(self, t: float) -> Pose:
        x, y, z = np.cos(t*0.5)*2, np.sin(t*0.5)*2, 0.5
        yaw = t * 0.5
        return Pose(
            position=np.array([x, y, z]),
            quaternion=np.array([np.cos(yaw/2), 0, 0, np.sin(yaw/2)]),
            timestamp=t
        )
    
    @staticmethod
    def _rot_to_quat(R):
        tr = np.trace(R)
        if tr > 0:
            s = 0.5 / np.sqrt(tr + 1)
            return np.array([0.25/s, (R[2,1]-R[1,2])*s, (R[0,2]-R[2,0])*s, (R[1,0]-R[0,1])*s])
        return np.array([1, 0, 0, 0])
    
    def get_map(self) -> Dict:
        if not self.slam:
            return {'keyframes': np.array([]), 'map_points': np.array([])}
        try:
            kfs = self.slam.GetAllKeyFrames()
            pts = self.slam.GetAllMapPoints()
            return {
                'keyframes': np.array([kf.GetPose() for kf in kfs]),
                'map_points': np.array([p.GetWorldPos() for p in pts])
            }
        except:
            return {'keyframes': np.array([]), 'map_points': np.array([])}
    
    def reset(self):
        if self.slam:
            self.slam.Reset()
        with self.imu_lock:
            self.imu_buffer.clear()
        self.current_pose = None
        self.tracking_status = TrackingStatus.NOT_INITIALIZED
    
    def save_map(self, path: str):
        if self.slam:
            self.slam.SaveMap(path)
    
    def load_map(self, path: str):
        if self.slam:
            self.slam.LoadMap(path)
    
    def shutdown(self):
        if self.slam:
            self.slam.Shutdown()