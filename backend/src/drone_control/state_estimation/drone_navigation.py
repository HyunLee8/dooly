"""
Complete Drone Navigation System
Integrates: Drone Control + IMU + SLAM + EKF Fusion
"""
import numpy as np
import time
from threading import Thread
from typing import Optional

# Drone control and sensors
from drone_control import TelloController, IMUReader
from drone_control.state_estimation import QuaternionFilter

# SLAM
from drone_control.slam import SLAMIntegration, CameraIntrinsics, TrackingStatus

# Sensor fusion
from sensor_fusion import SensorFusion, EKFConfig


class DroneNavigationSystem:
    """
    Complete navigation system with sensor fusion
    
    Architecture:
        IMU (200Hz) ──┐
                      ├──> EKF ──> Fused Pose (200Hz)
        SLAM (30Hz) ──┘
    """
    
    def __init__(self, enable_visualization: bool = False):
        """
        Initialize complete navigation system
        
        Args:
            enable_visualization: Show camera feed and pose visualization
        """
        # Drone controller
        self.controller = TelloController()
        self.imu_reader = IMUReader(self.controller.drone, buffer_size=200)
        
        # Camera calibration
        self.camera_intrinsics = CameraIntrinsics(
            fx=921.17, fy=919.02,
            cx=459.90, cy=351.24,
            k1=-0.033, k2=0.012
        )
        
        # SLAM system
        self.slam = SLAMIntegration(
            camera_intrinsics=self.camera_intrinsics
        )
        
        # EKF configuration optimized for drone
        ekf_config = EKFConfig(
            position_process_noise=0.01,
            velocity_process_noise=0.1,
            quaternion_process_noise=0.001,
            slam_position_noise=0.05,
            slam_quaternion_noise=0.01,
            gyro_noise=1.7e-4,
            accel_noise=2.0e-3,
        )
        
        # Sensor fusion
        self.fusion = SensorFusion(
            ekf_config=ekf_config,
            imu_rate_hz=200.0,
            enable_logging=True
        )
        
        self.enable_viz = enable_visualization
        self.running = False
        
        # Processing threads
        self.camera_thread = None
        self.imu_thread = None
        self.slam_thread = None
        
        print("[Navigation] System initialized")
    
    def connect(self) -> bool:
        """Connect to drone and initialize sensors"""
        if not self.controller.connect():
            print("[Navigation] Failed to connect to drone")
            return False
        
        # Start video stream
        self.controller.drone.streamon()
        time.sleep(1)
        
        print("[Navigation] Connected successfully")
        return True
    
    def start(self):
        """Start all navigation systems"""
        if self.running:
            return
        
        self.running = True
        
        # Start sensor fusion
        self.fusion.start()
        
        # Start SLAM
        self.slam.start()
        
        # Start IMU reader
        self.imu_reader.start_recording()
        
        # Start processing threads
        self.camera_thread = Thread(target=self._camera_loop, daemon=True)
        self.imu_thread = Thread(target=self._imu_loop, daemon=True)
        self.slam_thread = Thread(target=self._slam_fusion_loop, daemon=True)
        
        self.camera_thread.start()
        self.imu_thread.start()
        self.slam_thread.start()
        
        print("[Navigation] All systems started")
    
    def stop(self):
        """Stop all navigation systems"""
        if not self.running:
            return
        
        self.running = False
        
        # Stop threads
        if self.camera_thread:
            self.camera_thread.join(timeout=2)
        if self.imu_thread:
            self.imu_thread.join(timeout=2)
        if self.slam_thread:
            self.slam_thread.join(timeout=2)
        
        # Stop systems
        self.fusion.stop()
        self.slam.stop()
        self.imu_reader.stop_recording()
        
        print("[Navigation] All systems stopped")
    
    def _camera_loop(self):
        """Camera frame processing loop"""
        print("[Navigation] Camera loop started")
        
        while self.running:
            try:
                frame = self.controller.drone.get_frame_read().frame
                if frame is None:
                    time.sleep(0.01)
                    continue
                
                timestamp = time.time()
                self.slam.add_camera_frame(frame, timestamp)
                
                time.sleep(0.033)  # ~30 Hz
                
            except Exception as e:
                print(f"[Navigation] Camera error: {e}")
                time.sleep(0.1)
    
    def _imu_loop(self):
        """IMU data processing loop"""
        print("[Navigation] IMU loop started")
        
        while self.running:
            try:
                # Read IMU
                self.imu_reader.update()
                data = self.imu_reader.get_latest_data()
                
                if data:
                    # Send to SLAM
                    self.slam.add_imu_data(
                        accel=data['accel'],
                        gyro=data['gyro'],
                        timestamp=data['timestamp']
                    )
                    
                    # Send to fusion
                    self.fusion.add_imu(
                        accel=data['accel'],
                        gyro=data['gyro'],
                        timestamp=data['timestamp']
                    )
                
                time.sleep(0.005)  # 200 Hz
                
            except Exception as e:
                print(f"[Navigation] IMU error: {e}")
                time.sleep(0.01)
    
    def _slam_fusion_loop(self):
        """SLAM to fusion bridge loop"""
        print("[Navigation] SLAM fusion loop started")
        
        while self.running:
            try:
                # Get SLAM pose
                slam_pose, status = self.slam.get_pose()
                
                if slam_pose and status == TrackingStatus.OK:
                    # Send to fusion
                    self.fusion.add_slam(
                        position=slam_pose.position,
                        quaternion=slam_pose.quaternion,
                        timestamp=slam_pose.timestamp,
                        tracking_ok=True,
                        confidence=1.0
                    )
                elif status == TrackingStatus.LOST:
                    # SLAM lost, inform fusion with low confidence
                    if slam_pose:
                        self.fusion.add_slam(
                            position=slam_pose.position,
                            quaternion=slam_pose.quaternion,
                            timestamp=slam_pose.timestamp,
                            tracking_ok=False,
                            confidence=0.1
                        )
                
                time.sleep(0.033)  # ~30 Hz
                
            except Exception as e:
                print(f"[Navigation] SLAM fusion error: {e}")
                time.sleep(0.1)
    
    def get_pose(self) -> tuple:
        """
        Get current fused pose estimate
        
        Returns:
            (position, quaternion, covariance)
        """
        return self.fusion.get_pose()
    
    def get_position(self) -> np.ndarray:
        """Get current 3D position [x, y, z]"""
        return self.fusion.get_position()
    
    def get_velocity(self) -> np.ndarray:
        """Get current velocity [vx, vy, vz]"""
        return self.fusion.get_velocity()
    
    def get_orientation(self) -> np.ndarray:
        """Get current orientation [qw, qx, qy, qz]"""
        return self.fusion.get_orientation()
    
    def is_ready(self) -> bool:
        """Check if navigation system is ready"""
        return self.fusion.is_slam_available()
    
    def reset_navigation(self):
        """Reset navigation to current position"""
        pos, quat, _ = self.get_pose()
        self.fusion.reset(position=pos, orientation=quat)
        print("[Navigation] Reset at current position")
    
    def disconnect(self):
        """Disconnect and cleanup"""
        self.stop()
        self.controller.drone.streamoff()
        self.controller.disconnect()
        print("[Navigation] Disconnected")


def waypoint_navigation_demo(nav: DroneNavigationSystem):
    """
    Demo: Navigate to waypoints using fused pose
    """
    print("\n=== Waypoint Navigation Demo ===\n")
    
    # Wait for system to stabilize
    print("Waiting for navigation system to stabilize...")
    for _ in range(100):
        if nav.is_ready():
            print("Navigation ready!")
            break
        time.sleep(0.1)
    
    # Record start position
    start_pos = nav.get_position()
    start_quat = nav.get_orientation()
    print(f"Start position: {start_pos}")
    print(f"Start orientation: {start_quat}")
    
    # Takeoff
    print("\nTaking off...")
    nav.controller.takeoff()
    time.sleep(3)
    
    # Reset navigation at hover position
    nav.reset_navigation()
    hover_pos = nav.get_position()
    print(f"Hover position: {hover_pos}")
    
    # Define waypoints (relative to hover position)
    waypoints = [
        ("Forward 1m", np.array([1.0, 0.0, 0.0])),
        ("Right 1m", np.array([1.0, 1.0, 0.0])),
        ("Back 1m", np.array([0.0, 1.0, 0.0])),
        ("Left 1m", np.array([0.0, 0.0, 0.0])),
    ]
    
    for i, (desc, target_rel) in enumerate(waypoints):
        print(f"\n[Waypoint {i+1}] {desc}")
        target_pos = hover_pos + target_rel
        
        # Navigate to waypoint
        navigate_to_position(nav, target_pos, tolerance=0.2)
        
        # Check arrival
        final_pos = nav.get_position()
        error = np.linalg.norm(final_pos - target_pos)
        print(f"  Target: {target_pos}")
        print(f"  Actual: {final_pos}")
        print(f"  Error: {error:.3f}m")
    
    # Return to hover position
    print("\nReturning to start...")
    navigate_to_position(nav, hover_pos, tolerance=0.15)
    
    # Land
    print("\nLanding...")
    nav.controller.land()
    
    # Final statistics
    final_pos = nav.get_position()
    drift = np.linalg.norm(final_pos - hover_pos)
    print(f"\nFinal drift: {drift:.3f}m")


def navigate_to_position(
    nav: DroneNavigationSystem,
    target: np.ndarray,
    tolerance: float = 0.2,
    max_iterations: int = 20
):
    """
    Navigate drone to target position using pose feedback
    
    Args:
        nav: Navigation system
        target: Target position [x, y, z]
        tolerance: Position tolerance in meters
        max_iterations: Maximum control iterations
    """
    for iteration in range(max_iterations):
        current_pos = nav.get_position()
        error = target - current_pos
        distance = np.linalg.norm(error)
        
        if distance < tolerance:
            print(f"  Reached target (error: {distance:.3f}m)")
            return
        
        # Simple proportional control
        # Scale movement by error (max 50cm per command)
        move_dist = min(distance * 100, 50)  # cm
        
        # Determine primary direction
        direction = error / distance
        
        if abs(error[0]) > abs(error[1]):
            # Move forward/back
            if error[0] > 0:
                nav.controller.move_forward(int(move_dist))
            else:
                nav.controller.move_back(int(move_dist))
        else:
            # Move left/right
            if error[1] > 0:
                nav.controller.move_right(int(move_dist))
            else:
                nav.controller.move_left(int(move_dist))
        
        time.sleep(1.5)
    
    print(f"  Max iterations reached (error: {distance:.3f}m)")


def main():
    """Main navigation demo"""
    nav = DroneNavigationSystem(enable_visualization=False)
    
    try:
        # Connect
        if not nav.connect():
            return
        
        # Start navigation
        nav.start()
        
        # Wait for initialization
        time.sleep(3)
        
        # Run waypoint demo
        waypoint_navigation_demo(nav)
        
        # Monitor for a bit
        print("\nMonitoring navigation...")
        for i in range(50):
            pos = nav.get_position()
            vel = nav.get_velocity()
            
            if i % 10 == 0:
                print(f"\nPosition: {pos}")
                print(f"Velocity: {vel}")
                print(f"SLAM available: {nav.is_ready()}")
            
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("\nShutting down...")
        nav.disconnect()
        print("Complete!")


if __name__ == "__main__":
    main()