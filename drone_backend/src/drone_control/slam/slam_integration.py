"""
SLAM Integration Module for Drone Navigation
Combines camera frames, IMU data, and ORB-SLAM3 for pose estimation
"""
import numpy as np
import cv2
import time
from typing import Optional, Tuple
from threading import Thread, Lock
from queue import Queue, Empty
from dataclasses import dataclass

from .orbslam3_wrapper import ORBSLAM3VIO, IMUMeasurement, Pose, CameraIntrinsics, TrackingStatus


@dataclass
class SLAMConfig:
    """SLAM configuration"""
    vocab_path: str = "ORBvoc.txt"
    config_path: str = "TelloVIO.yaml"
    max_imu_queue_size: int = 1000
    max_frame_queue_size: int = 30
    process_frequency_hz: float = 30.0
    

class SLAMIntegration:
    """
    High-level SLAM integration for drone navigation
    Handles asynchronous frame/IMU processing and provides clean API
    """
    
    def __init__(self, camera_intrinsics: CameraIntrinsics, config: Optional[SLAMConfig] = None):
        """
        Initialize SLAM integration
        
        Args:
            camera_intrinsics: Camera calibration parameters
            config: SLAM configuration (uses defaults if None)
        """
        self.config = config or SLAMConfig()
        self.camera_intrinsics = camera_intrinsics
        
        # Initialize ORB-SLAM3
        self.slam = ORBSLAM3VIO(
            vocab_path=self.config.vocab_path,
            config_path=self.config.config_path,
            camera_intrinsics=camera_intrinsics,
            verbose=True
        )
        
        # Processing queues
        self.imu_queue = Queue(maxsize=self.config.max_imu_queue_size)
        self.frame_queue = Queue(maxsize=self.config.max_frame_queue_size)
        
        # State
        self.current_pose: Optional[Pose] = None
        self.tracking_status = TrackingStatus.NOT_INITIALIZED
        self.pose_lock = Lock()
        
        # Processing thread
        self.running = False
        self.process_thread: Optional[Thread] = None
        
        # Statistics
        self.total_frames = 0
        self.lost_tracking_count = 0
        self.start_time = time.time()
        
        print("[SLAM Integration] Initialized")
        print(f"  Camera: fx={camera_intrinsics.fx:.1f}, fy={camera_intrinsics.fy:.1f}")
        print(f"  Target frequency: {self.config.process_frequency_hz} Hz")
    
    def start(self):
        """Start SLAM processing thread"""
        if self.running:
            print("[SLAM Integration] Already running")
            return
        
        self.running = True
        self.process_thread = Thread(target=self._processing_loop, daemon=True)
        self.process_thread.start()
        print("[SLAM Integration] Processing started")
    
    def stop(self):
        """Stop SLAM processing"""
        if not self.running:
            return
        
        self.running = False
        if self.process_thread:
            self.process_thread.join(timeout=2.0)
        
        print("[SLAM Integration] Processing stopped")
        self._print_statistics()
    
    def add_camera_frame(self, frame: np.ndarray, timestamp: float):
        """
        Add camera frame for processing
        
        Args:
            frame: RGB or grayscale image
            timestamp: Frame timestamp in seconds
        """
        try:
            self.frame_queue.put_nowait((frame, timestamp))
        except:
            # Queue full, drop oldest
            try:
                self.frame_queue.get_nowait()
                self.frame_queue.put_nowait((frame, timestamp))
            except:
                pass
    
    def add_imu_data(self, accel: np.ndarray, gyro: np.ndarray, timestamp: float):
        """
        Add IMU measurement
        
        Args:
            accel: Acceleration [ax, ay, az] in m/s^2
            gyro: Angular velocity [gx, gy, gz] in rad/s
            timestamp: Measurement timestamp in seconds
        """
        imu = IMUMeasurement(timestamp=timestamp, accel=accel, gyro=gyro)
        
        try:
            self.imu_queue.put_nowait(imu)
        except:
            # Queue full, drop oldest
            try:
                self.imu_queue.get_nowait()
                self.imu_queue.put_nowait(imu)
            except:
                pass
    
    def get_pose(self) -> Tuple[Optional[Pose], TrackingStatus]:
        """
        Get current pose estimate
        
        Returns:
            (pose, tracking_status): Latest pose and tracking status
        """
        with self.pose_lock:
            return self.current_pose, self.tracking_status
    
    def get_position(self) -> Optional[np.ndarray]:
        """Get current 3D position [x, y, z]"""
        with self.pose_lock:
            return self.current_pose.position if self.current_pose else None
    
    def get_orientation(self) -> Optional[np.ndarray]:
        """Get current orientation quaternion [qw, qx, qy, qz]"""
        with self.pose_lock:
            return self.current_pose.quaternion if self.current_pose else None
    
    def is_tracking(self) -> bool:
        """Check if SLAM is currently tracking"""
        with self.pose_lock:
            return self.tracking_status == TrackingStatus.OK
    
    def get_map_visualization(self) -> dict:
        """
        Get map data for visualization
        
        Returns:
            Dictionary with 'keyframes' and 'map_points'
        """
        return self.slam.get_map()
    
    def reset(self):
        """Reset SLAM system"""
        # Clear queues
        while not self.imu_queue.empty():
            try:
                self.imu_queue.get_nowait()
            except Empty:
                break
        
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except Empty:
                break
        
        # Reset SLAM
        self.slam.reset()
        
        with self.pose_lock:
            self.current_pose = None
            self.tracking_status = TrackingStatus.NOT_INITIALIZED
        
        print("[SLAM Integration] System reset")
    
    def save_map(self, filepath: str):
        """Save current map to file"""
        self.slam.save_map(filepath)
        print(f"[SLAM Integration] Map saved to {filepath}")
    
    def load_map(self, filepath: str):
        """Load map from file"""
        self.slam.load_map(filepath)
        print(f"[SLAM Integration] Map loaded from {filepath}")
    
    def _processing_loop(self):
        """Main processing loop (runs in separate thread)"""
        print("[SLAM Integration] Processing loop started")
        
        frame_dt = 1.0 / self.config.process_frequency_hz
        
        while self.running:
            loop_start = time.time()
            
            try:
                # Get frame (block with timeout)
                try:
                    frame, frame_time = self.frame_queue.get(timeout=0.1)
                except Empty:
                    continue
                
                # Collect IMU measurements
                imu_measurements = []
                while not self.imu_queue.empty():
                    try:
                        imu = self.imu_queue.get_nowait()
                        self.slam.add_imu_measurement(imu)
                        imu_measurements.append(imu)
                    except Empty:
                        break
                
                # Process frame
                pose, status = self.slam.process_frame(frame, frame_time)
                
                # Update state
                with self.pose_lock:
                    self.current_pose = pose
                    self.tracking_status = status
                
                self.total_frames += 1
                
                if status == TrackingStatus.LOST:
                    self.lost_tracking_count += 1
                
                # Rate limiting
                elapsed = time.time() - loop_start
                if elapsed < frame_dt:
                    time.sleep(frame_dt - elapsed)
                
            except Exception as e:
                print(f"[SLAM Integration] Processing error: {e}")
                time.sleep(0.1)
        
        print("[SLAM Integration] Processing loop ended")
    
    def _print_statistics(self):
        """Print processing statistics"""
        runtime = time.time() - self.start_time
        avg_fps = self.total_frames / runtime if runtime > 0 else 0
        tracking_rate = (self.total_frames - self.lost_tracking_count) / self.total_frames * 100 if self.total_frames > 0 else 0
        
        print("\n=== SLAM Statistics ===")
        print(f"Total frames: {self.total_frames}")
        print(f"Runtime: {runtime:.1f}s")
        print(f"Average FPS: {avg_fps:.1f}")
        print(f"Tracking rate: {tracking_rate:.1f}%")
        print(f"Lost tracking: {self.lost_tracking_count} times")
    
    def shutdown(self):
        """Shutdown SLAM system"""
        self.stop()
        self.slam.shutdown()
        print("[SLAM Integration] Shutdown complete")


# Example usage with drone
if __name__ == "__main__":
    # Camera calibration for DJI Tello
    intrinsics = CameraIntrinsics(
        fx=921.17, fy=919.02,
        cx=459.90, cy=351.24,
        k1=-0.033, k2=0.012
    )
    
    # Initialize SLAM
    slam_integration = SLAMIntegration(intrinsics)
    slam_integration.start()
    
    # Simulate drone flight
    try:
        for i in range(300):  # 10 seconds at 30 Hz
            # Generate fake camera frame
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            frame_time = i / 30.0
            
            # Add IMU data (10x camera rate)
            for j in range(10):
                imu_time = frame_time + j / 300.0
                accel = np.array([0.0, 0.0, 9.81]) + np.random.randn(3) * 0.1
                gyro = np.array([0.0, 0.0, 0.1]) + np.random.randn(3) * 0.01
                slam_integration.add_imu_data(accel, gyro, imu_time)
            
            # Add camera frame
            slam_integration.add_camera_frame(frame, frame_time)
            
            # Get pose
            pose, status = slam_integration.get_pose()
            
            if pose and i % 30 == 0:
                print(f"\nFrame {i}: Status={status.name}")
                print(f"  Position: {pose.position}")
                print(f"  Quaternion: {pose.quaternion}")
            
            time.sleep(0.033)  # 30 Hz
    
    except KeyboardInterrupt:
        print("\nInterrupted")
    
    finally:
        slam_integration.shutdown()