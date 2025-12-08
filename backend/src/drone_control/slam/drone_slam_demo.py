"""
Complete Drone SLAM Demo
Integrates DJI Tello with ORB-SLAM3 for visual-inertial odometry
"""
import numpy as np
import cv2
import time
from threading import Thread

# Import drone control
from drone_control import TelloController, IMUReader

# Import SLAM
from slam_integration import SLAMIntegration, CameraIntrinsics, SLAMConfig


class DroneSLAM:
    """
    Complete drone SLAM system
    Combines drone control, IMU reading, camera streaming, and SLAM
    """
    
    def __init__(self, enable_visualization: bool = True):
        """
        Initialize drone SLAM system
        
        Args:
            enable_visualization: Show camera feed and tracking info
        """
        # Initialize drone
        self.controller = TelloController()
        self.imu_reader = IMUReader(self.controller.drone, buffer_size=200)
        
        # Camera calibration for Tello
        self.camera_intrinsics = CameraIntrinsics(
            fx=921.17, fy=919.02,
            cx=459.90, cy=351.24,
            k1=-0.033, k2=0.012
        )
        
        # SLAM configuration
        slam_config = SLAMConfig(
            vocab_path="ORBvoc.txt",
            config_path="TelloVIO.yaml",
            process_frequency_hz=30.0
        )
        
        # Initialize SLAM
        self.slam = SLAMIntegration(self.camera_intrinsics, slam_config)
        
        self.enable_viz = enable_visualization
        self.running = False
        
        # Threads
        self.camera_thread = None
        self.imu_thread = None
        
        print("[Drone SLAM] Initialized")
    
    def connect(self) -> bool:
        """Connect to drone"""
        if not self.controller.connect():
            print("[Drone SLAM] Failed to connect to drone")
            return False
        
        # Start video stream
        self.controller.drone.streamon()
        time.sleep(1)
        
        print("[Drone SLAM] Connected successfully")
        return True
    
    def start(self):
        """Start SLAM system"""
        if self.running:
            return
        
        self.running = True
        
        # Start SLAM processing
        self.slam.start()
        
        # Start IMU reading
        self.imu_reader.start_recording()
        
        # Start camera and IMU threads
        self.camera_thread = Thread(target=self._camera_loop, daemon=True)
        self.imu_thread = Thread(target=self._imu_loop, daemon=True)
        
        self.camera_thread.start()
        self.imu_thread.start()
        
        print("[Drone SLAM] Started")
    
    def stop(self):
        """Stop SLAM system"""
        if not self.running:
            return
        
        self.running = False
        
        # Stop threads
        if self.camera_thread:
            self.camera_thread.join(timeout=2)
        if self.imu_thread:
            self.imu_thread.join(timeout=2)
        
        # Stop systems
        self.slam.stop()
        self.imu_reader.stop_recording()
        
        print("[Drone SLAM] Stopped")
    
    def _camera_loop(self):
        """Camera streaming loop"""
        print("[Drone SLAM] Camera thread started")
        
        frame_count = 0
        
        while self.running:
            try:
                # Get frame from Tello
                frame = self.controller.drone.get_frame_read().frame
                
                if frame is None:
                    time.sleep(0.01)
                    continue
                
                # Get timestamp
                timestamp = time.time()
                
                # Send to SLAM
                self.slam.add_camera_frame(frame, timestamp)
                
                # Visualization
                if self.enable_viz:
                    self._visualize_frame(frame, timestamp)
                
                frame_count += 1
                
                # Limit to ~30 FPS
                time.sleep(0.033)
                
            except Exception as e:
                print(f"[Drone SLAM] Camera error: {e}")
                time.sleep(0.1)
        
        print(f"[Drone SLAM] Camera thread ended ({frame_count} frames)")
    
    def _imu_loop(self):
        """IMU reading loop"""
        print("[Drone SLAM] IMU thread started")
        
        sample_count = 0
        
        while self.running:
            try:
                # Update IMU
                self.imu_reader.update()
                data = self.imu_reader.get_latest_data()
                
                if data:
                    # Send to SLAM
                    self.slam.add_imu_data(
                        accel=data['accel'],
                        gyro=data['gyro'],
                        timestamp=data['timestamp']
                    )
                    
                    sample_count += 1
                
                # Run at ~200 Hz
                time.sleep(0.005)
                
            except Exception as e:
                print(f"[Drone SLAM] IMU error: {e}")
                time.sleep(0.01)
        
        print(f"[Drone SLAM] IMU thread ended ({sample_count} samples)")
    
    def _visualize_frame(self, frame: np.ndarray, timestamp: float):
        """Visualize camera frame with SLAM info"""
        viz_frame = frame.copy()
        
        # Get SLAM pose
        pose, status = self.slam.get_pose()
        
        # Draw status
        status_text = f"Status: {status.name}"
        color = (0, 255, 0) if status.name == 'OK' else (0, 0, 255)
        cv2.putText(viz_frame, status_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Draw pose if available
        if pose:
            pos_text = f"Pos: ({pose.position[0]:.2f}, {pose.position[1]:.2f}, {pose.position[2]:.2f})"
            cv2.putText(viz_frame, pos_text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Draw battery
        battery = self.controller.get_battery()
        battery_text = f"Battery: {battery}%"
        cv2.putText(viz_frame, battery_text, (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Show frame
        cv2.imshow('Drone SLAM', viz_frame)
        cv2.waitKey(1)
    
    def get_pose(self):
        """Get current drone pose from SLAM"""
        return self.slam.get_pose()
    
    def get_position(self):
        """Get current 3D position"""
        return self.slam.get_position()
    
    def is_tracking(self):
        """Check if SLAM is tracking"""
        return self.slam.is_tracking()
    
    def save_map(self, filepath: str):
        """Save SLAM map"""
        self.slam.save_map(filepath)
    
    def load_map(self, filepath: str):
        """Load SLAM map"""
        self.slam.load_map(filepath)
    
    def reset_slam(self):
        """Reset SLAM system"""
        self.slam.reset()
    
    def disconnect(self):
        """Disconnect from drone"""
        self.stop()
        self.controller.drone.streamoff()
        self.controller.disconnect()
        cv2.destroyAllWindows()
        print("[Drone SLAM] Disconnected")


def autonomous_flight_demo(drone_slam: DroneSLAM):
    """
    Demo: Autonomous flight using SLAM localization
    """
    print("\n=== Autonomous Flight Demo ===\n")
    
    # Wait for SLAM initialization
    print("Waiting for SLAM to initialize...")
    for _ in range(50):
        if drone_slam.is_tracking():
            print("SLAM initialized!")
            break
        time.sleep(0.1)
    else:
        print("Warning: SLAM not tracking, continuing anyway...")
    
    # Takeoff
    print("\nTaking off...")
    drone_slam.controller.takeoff()
    time.sleep(3)
    
    # Record initial position
    start_pos = drone_slam.get_position()
    if start_pos is not None:
        print(f"Start position: {start_pos}")
    
    # Fly in square while monitoring position
    movements = [
        ("Forward 50cm", lambda: drone_slam.controller.move_forward(50)),
        ("Right 50cm", lambda: drone_slam.controller.move_right(50)),
        ("Back 50cm", lambda: drone_slam.controller.move_back(50)),
        ("Left 50cm", lambda: drone_slam.controller.move_left(50)),
    ]
    
    for desc, move_fn in movements:
        print(f"\n{desc}")
        
        # Get position before
        pos_before = drone_slam.get_position()
        
        # Execute movement
        move_fn()
        time.sleep(2)
        
        # Get position after
        pos_after = drone_slam.get_position()
        
        if pos_before is not None and pos_after is not None:
            distance = np.linalg.norm(pos_after - pos_before)
            print(f"  SLAM measured distance: {distance:.2f}m")
            print(f"  Position: {pos_after}")
        
        # Check tracking
        if not drone_slam.is_tracking():
            print("  WARNING: Lost tracking!")
    
    # Return and land
    print("\nLanding...")
    drone_slam.controller.land()
    
    # Final position
    final_pos = drone_slam.get_position()
    if start_pos is not None and final_pos is not None:
        drift = np.linalg.norm(final_pos - start_pos)
        print(f"\nTotal drift: {drift:.2f}m")


def main():
    """Main demo"""
    drone_slam = DroneSLAM(enable_visualization=True)
    
    try:
        # Connect
        if not drone_slam.connect():
            return
        
        # Start SLAM
        drone_slam.start()
        
        # Wait a moment for everything to initialize
        time.sleep(2)
        
        # Run autonomous flight
        autonomous_flight_demo(drone_slam)
        
        # Keep running for a bit
        print("\nMonitoring for 5 seconds...")
        time.sleep(5)
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        print("\nShutting down...")
        drone_slam.disconnect()
        print("Done!")


if __name__ == "__main__":
    main()