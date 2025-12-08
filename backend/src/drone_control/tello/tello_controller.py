"""
Tello Controller - High-level integration of Tello + SLAM + ESKF
Provides unified interface for autonomous drone navigation
"""

import numpy as np
import time
from typing import Optional, Tuple
from enum import Enum
from threading import Thread, Lock

from .tello_interface import TelloInterface, TelloState
from ..slam.slam_integration import SLAMIntegration, SLAMConfig
from ..slam.orbslam3_wrapper import CameraIntrinsics, TrackingStatus
from ..state_estimation.imu_slam_fusion import IMUSLAMFusion, IMUData, SLAMPose, FusedPose


class FlightMode(Enum):
    """Flight modes"""
    IDLE = 0
    MANUAL = 1
    POSITION_HOLD = 2
    AUTONOMOUS = 3


class TelloController:
    """
    High-level Tello controller with SLAM and ESKF integration
    Combines drone control, visual SLAM, and sensor fusion
    """
    
    def __init__(self, enable_slam: bool = True, enable_eskf: bool = True):
        """
        Initialize Tello controller
        
        Args:
            enable_slam: Enable SLAM system
            enable_eskf: Enable ESKF sensor fusion
        """
        # Core components
        self.tello = TelloInterface()
        self.enable_slam = enable_slam
        self.enable_eskf = enable_eskf
        
        # SLAM system (if enabled)
        self.slam: Optional[SLAMIntegration] = None
        if enable_slam:
            # Tello camera calibration
            intrinsics = CameraIntrinsics(
                fx=921.17, fy=919.02,
                cx=459.90, cy=351.24,
                k1=-0.033, k2=0.012
            )
            
            slam_config = SLAMConfig(
                vocab_path="ORBvoc.txt",
                config_path="TelloVIO.yaml",
                process_frequency_hz=30.0
            )
            
            self.slam = SLAMIntegration(intrinsics, slam_config)
        
        # ESKF sensor fusion (if enabled)
        self.eskf: Optional[IMUSLAMFusion] = None
        if enable_eskf:
            self.eskf = IMUSLAMFusion(
                process_noise_acc=0.1,
                process_noise_gyro=0.01,
                process_noise_bias=0.001,
                measurement_noise_pos=0.05,
                measurement_noise_orient=0.02
            )
        
        # State
        self.flight_mode = FlightMode.IDLE
        self.fused_pose: Optional[FusedPose] = None
        self.pose_lock = Lock()
        
        # Processing
        self.running = False
        self.process_thread: Optional[Thread] = None
        
        # Statistics
        self.imu_update_count = 0
        self.slam_update_count = 0
        self.start_time = None
        
        print("[Tello Controller] Initialized")
        print(f"  SLAM: {'Enabled' if enable_slam else 'Disabled'}")
        print(f"  ESKF: {'Enabled' if enable_eskf else 'Disabled'}")
    
    def connect(self) -> bool:
        """
        Connect to Tello and start systems
        
        Returns:
            True if successful
        """
        print("[Tello Controller] Connecting...")
        
        if not self.tello.connect():
            return False
        
        # Start video streaming
        self.tello.start_video_stream(frame_callback=self._on_frame)
        
        # Start SLAM if enabled
        if self.slam:
            self.slam.start()
        
        # Start processing loop
        self.start_time = time.time()
        self.running = True
        self.process_thread = Thread(target=self._processing_loop, daemon=True)
        self.process_thread.start()
        
        print("[Tello Controller] Connected and ready")
        return True
    
    def disconnect(self):
        """Disconnect and shutdown all systems"""
        print("[Tello Controller] Disconnecting...")
        
        self.running = False
        if self.process_thread:
            self.process_thread.join(timeout=2.0)
        
        if self.slam:
            self.slam.shutdown()
        
        self.tello.disconnect()
        
        self._print_statistics()
        print("[Tello Controller] Disconnected")
    
    # ========== Flight Commands ==========
    
    def takeoff(self) -> bool:
        """Takeoff and switch to position hold mode"""
        if self.tello.takeoff():
            self.flight_mode = FlightMode.POSITION_HOLD
            return True
        return False
    
    def land(self) -> bool:
        """Land and switch to idle mode"""
        if self.tello.land():
            self.flight_mode = FlightMode.IDLE
            return True
        return False
    
    def emergency_stop(self):
        """Emergency stop all motors"""
        self.tello.emergency()
        self.flight_mode = FlightMode.IDLE
    
    def set_mode(self, mode: FlightMode):
        """Set flight mode"""
        self.flight_mode = mode
        print(f"[Tello Controller] Mode: {mode.name}")
    
    def move_relative(self, x: float, y: float, z: float, speed: int = 30):
        """
        Move relative to current position (in meters)
        
        Args:
            x: Forward/backward in meters
            y: Left/right in meters
            z: Up/down in meters
            speed: Speed in cm/s (10-100)
        """
        # Convert meters to cm
        x_cm = int(x * 100)
        y_cm = int(y * 100)
        z_cm = int(z * 100)
        
        self.tello.move(x_cm, y_cm, z_cm, speed)
    
    def rotate(self, degrees: int):
        """Rotate by degrees (positive = clockwise)"""
        self.tello.rotate(degrees)
    
    def send_velocity_command(self, vx: float, vy: float, vz: float, yaw_rate: float):
        """
        Send velocity commands (m/s)
        
        Args:
            vx: Forward/backward velocity (m/s, -1 to 1)
            vy: Left/right velocity (m/s, -1 to 1)
            vz: Up/down velocity (m/s, -1 to 1)
            yaw_rate: Yaw rate (rad/s, -1 to 1)
        """
        # Convert to Tello RC values (-100 to 100)
        lr = int(np.clip(vy * 100, -100, 100))
        fb = int(np.clip(vx * 100, -100, 100))
        ud = int(np.clip(vz * 100, -100, 100))
        yaw = int(np.clip(yaw_rate * 100, -100, 100))
        
        self.tello.send_rc_control(lr, fb, ud, yaw)
    
    # ========== State Queries ==========
    
    def get_fused_pose(self) -> Optional[FusedPose]:
        """Get fused pose from ESKF"""
        with self.pose_lock:
            return self.fused_pose
    
    def get_tello_state(self) -> Optional[TelloState]:
        """Get raw Tello state"""
        return self.tello.get_state()
    
    def get_slam_pose(self) -> Tuple[Optional, Optional]:
        """Get SLAM pose and tracking status"""
        if self.slam:
            return self.slam.get_pose()
        return None, None
    
    def get_battery(self) -> int:
        """Get battery percentage"""
        state = self.tello.get_state()
        return state.battery if state else 0
    
    def is_flying(self) -> bool:
        """Check if drone is flying"""
        state = self.tello.get_state()
        return state.is_flying if state else False
    
    def get_current_frame(self) -> Optional[np.ndarray]:
        """Get current camera frame"""
        return self.tello.get_frame()
    
    # ========== Advanced Functions ==========
    
    def save_slam_map(self, filepath: str):
        """Save SLAM map to file"""
        if self.slam:
            self.slam.save_map(filepath)
    
    def load_slam_map(self, filepath: str):
        """Load SLAM map from file"""
        if self.slam:
            self.slam.load_map(filepath)
    
    def reset_slam(self):
        """Reset SLAM system"""
        if self.slam:
            self.slam.reset()
    
    def reset_eskf(self):
        """Reset ESKF filter"""
        if self.eskf:
            self.eskf.reset()
    
    # ========== Internal Processing ==========
    
    def _processing_loop(self):
        """Main processing loop for IMU and state updates"""
        print("[Tello Controller] Processing loop started")
        
        last_imu_time = time.time()
        imu_dt = 1.0 / 100.0  # Target 100Hz IMU processing
        
        while self.running:
            loop_start = time.time()
            
            try:
                # Get Tello state (includes IMU-like data)
                tello_state = self.tello.get_state()
                
                if tello_state and self.eskf:
                    # Create IMU data from Tello state
                    # Note: Tello doesn't provide true IMU, this is estimated from state
                    current_time = tello_state.timestamp
                    
                    if current_time - last_imu_time >= imu_dt:
                        # Convert Tello acceleration to m/s^2
                        accel = tello_state.get_acceleration_mps2()
                        
                        # Estimate gyro from velocity changes (rough approximation)
                        gyro = np.array([0.0, 0.0, 0.0])  # Tello doesn't expose gyro
                        
                        imu_data = IMUData(
                            ax=accel[0], ay=accel[1], az=accel[2],
                            gx=gyro[0], gy=gyro[1], gz=gyro[2],
                            timestamp=current_time
                        )
                        
                        # ESKF prediction
                        fused = self.eskf.predict(imu_data)
                        
                        with self.pose_lock:
                            self.fused_pose = fused
                        
                        self.imu_update_count += 1
                        last_imu_time = current_time
                        
                        # Also send IMU to SLAM if enabled
                        if self.slam:
                            self.slam.add_imu_data(accel, gyro, current_time)
                
            except Exception as e:
                print(f"[Tello Controller] Processing error: {e}")
            
            # Rate limiting
            elapsed = time.time() - loop_start
            if elapsed < imu_dt:
                time.sleep(imu_dt - elapsed)
        
        print("[Tello Controller] Processing loop ended")
    
    def _on_frame(self, frame: np.ndarray, timestamp: float):
        """Callback when new camera frame arrives"""
        # Send to SLAM
        if self.slam:
            self.slam.add_camera_frame(frame, timestamp)
        
        # Get SLAM pose and update ESKF
        if self.slam and self.eskf:
            slam_pose, tracking_status = self.slam.get_pose()
            
            if slam_pose:
                # Convert to ESKF format
                eskf_slam_pose = SLAMPose(
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
                
                # ESKF update
                fused = self.eskf.update(eskf_slam_pose)
                
                with self.pose_lock:
                    self.fused_pose = fused
                
                self.slam_update_count += 1
    
    def _print_statistics(self):
        """Print processing statistics"""
        if not self.start_time:
            return
        
        runtime = time.time() - self.start_time
        
        print("\n=== Tello Controller Statistics ===")
        print(f"Runtime: {runtime:.1f}s")
        print(f"IMU updates: {self.imu_update_count} ({self.imu_update_count/runtime:.1f} Hz)")
        print(f"SLAM updates: {self.slam_update_count} ({self.slam_update_count/runtime:.1f} Hz)")


# Example usage
if __name__ == "__main__":
    # Initialize controller with SLAM and ESKF
    controller = TelloController(enable_slam=True, enable_eskf=True)
    
    if controller.connect():
        try:
            print("\nBattery:", controller.get_battery(), "%")
            
            # Simple autonomous flight
            print("\nTaking off...")
            controller.takeoff()
            time.sleep(3)
            
            # Monitor fused pose
            for i in range(10):
                pose = controller.get_fused_pose()
                if pose:
                    print(f"Fused position: ({pose.x:.2f}, {pose.y:.2f}, {pose.z:.2f})")
                time.sleep(0.5)
            
            print("\nLanding...")
            controller.land()
            
        except KeyboardInterrupt:
            print("\nInterrupted - Emergency landing")
            controller.land()
        
        finally:
            controller.disconnect()