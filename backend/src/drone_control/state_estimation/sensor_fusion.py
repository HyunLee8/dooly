"""
Sensor Fusion Module
Combines IMU, SLAM, and EKF for robust pose estimation
"""
import numpy as np
from typing import Optional, Tuple
from threading import Thread, Lock
from queue import Queue, Empty
import time

from kalman_filter import (
    ExtendedKalmanFilter,
    EKFState,
    EKFConfig,
    IMUData,
    SLAMPose
)


class SensorFusion:
    """
    High-level sensor fusion system
    Manages IMU and SLAM data streams and provides fused pose output
    """
    
    def __init__(
        self,
        initial_position: np.ndarray = None,
        initial_orientation: np.ndarray = None,
        ekf_config: Optional[EKFConfig] = None,
        imu_rate_hz: float = 200.0,
        enable_logging: bool = False
    ):
        """
        Initialize sensor fusion
        
        Args:
            initial_position: Initial position [x, y, z] (default: origin)
            initial_orientation: Initial quaternion [qw, qx, qy, qz] (default: identity)
            ekf_config: EKF configuration
            imu_rate_hz: Expected IMU update rate
            enable_logging: Enable state logging for debugging
        """
        # Initial state
        if initial_position is None:
            initial_position = np.zeros(3)
        if initial_orientation is None:
            initial_orientation = np.array([1, 0, 0, 0])
        
        initial_state = EKFState(
            position=initial_position,
            velocity=np.zeros(3),
            quaternion=initial_orientation,
            gyro_bias=np.zeros(3),
            accel_bias=np.zeros(3),
            timestamp=time.time()
        )
        
        # Initialize EKF
        self.ekf = ExtendedKalmanFilter(initial_state, ekf_config)
        
        # Queues for async processing
        self.imu_queue = Queue(maxsize=1000)
        self.slam_queue = Queue(maxsize=10)
        
        # State access lock
        self.state_lock = Lock()
        
        # Current fused state
        self.fused_state: Optional[EKFState] = initial_state.copy()
        self.fused_covariance: Optional[np.ndarray] = None
        
        # Processing control
        self.running = False
        self.process_thread: Optional[Thread] = None
        
        # Configuration
        self.imu_rate_hz = imu_rate_hz
        self.enable_logging = enable_logging
        
        # Logging
        if enable_logging:
            self.log_file = open('sensor_fusion_log.txt', 'w')
            self.log_file.write("timestamp,px,py,pz,vx,vy,vz,qw,qx,qy,qz,slam_update\n")
        else:
            self.log_file = None
        
        # Statistics
        self.imu_count = 0
        self.slam_count = 0
        self.start_time = time.time()
        
        print("[Sensor Fusion] Initialized")
        print(f"  Initial position: {initial_position}")
        print(f"  IMU rate: {imu_rate_hz} Hz")
    
    def start(self):
        """Start sensor fusion processing"""
        if self.running:
            print("[Sensor Fusion] Already running")
            return
        
        self.running = True
        self.process_thread = Thread(target=self._processing_loop, daemon=True)
        self.process_thread.start()
        
        print("[Sensor Fusion] Processing started")
    
    def stop(self):
        """Stop sensor fusion processing"""
        if not self.running:
            return
        
        self.running = False
        if self.process_thread:
            self.process_thread.join(timeout=2.0)
        
        if self.log_file:
            self.log_file.close()
        
        print("[Sensor Fusion] Stopped")
        self._print_statistics()
    
    def add_imu(self, accel: np.ndarray, gyro: np.ndarray, timestamp: float):
        """
        Add IMU measurement
        
        Args:
            accel: Acceleration [ax, ay, az] m/s^2
            gyro: Angular velocity [gx, gy, gz] rad/s
            timestamp: Measurement time in seconds
        """
        imu = IMUData(
            accel=np.array(accel),
            gyro=np.array(gyro),
            timestamp=timestamp
        )
        
        try:
            self.imu_queue.put_nowait(imu)
        except:
            # Queue full, drop oldest
            try:
                self.imu_queue.get_nowait()
                self.imu_queue.put_nowait(imu)
            except:
                pass
    
    def add_slam(
        self,
        position: np.ndarray,
        quaternion: np.ndarray,
        timestamp: float,
        tracking_ok: bool = True,
        confidence: float = 1.0
    ):
        """
        Add SLAM pose measurement
        
        Args:
            position: Position [x, y, z] meters
            quaternion: Orientation [qw, qx, qy, qz]
            timestamp: Measurement time in seconds
            tracking_ok: Whether SLAM is tracking correctly
            confidence: Confidence in measurement (0-1)
        """
        slam = SLAMPose(
            position=np.array(position),
            quaternion=np.array(quaternion),
            timestamp=timestamp,
            tracking_ok=tracking_ok,
            confidence=confidence
        )
        
        try:
            self.slam_queue.put_nowait(slam)
        except:
            # Queue full, drop oldest
            try:
                self.slam_queue.get_nowait()
                self.slam_queue.put_nowait(slam)
            except:
                pass
    
    def get_pose(self) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Get current fused pose
        
        Returns:
            (position, quaternion, covariance):
                - position: [x, y, z] meters
                - quaternion: [qw, qx, qy, qz]
                - covariance: 16x16 matrix or None
        """
        with self.state_lock:
            if self.fused_state is None:
                return np.zeros(3), np.array([1, 0, 0, 0]), None
            
            return (
                self.fused_state.position.copy(),
                self.fused_state.quaternion.copy(),
                self.fused_covariance.copy() if self.fused_covariance is not None else None
            )
    
    def get_position(self) -> np.ndarray:
        """Get current position [x, y, z]"""
        pos, _, _ = self.get_pose()
        return pos
    
    def get_velocity(self) -> np.ndarray:
        """Get current velocity [vx, vy, vz]"""
        with self.state_lock:
            if self.fused_state is None:
                return np.zeros(3)
            return self.fused_state.velocity.copy()
    
    def get_orientation(self) -> np.ndarray:
        """Get current orientation [qw, qx, qy, qz]"""
        _, quat, _ = self.get_pose()
        return quat
    
    def get_orientation_rates(self) -> np.ndarray:
        """Get orientation rates [roll_rate, pitch_rate, yaw_rate]"""
        return self.ekf.get_orientation_rates()
    
    def get_full_state(self) -> Tuple[EKFState, np.ndarray]:
        """
        Get complete EKF state
        
        Returns:
            (state, covariance): Full state and covariance matrix
        """
        with self.state_lock:
            return self.ekf.get_pose()
    
    def is_slam_available(self) -> bool:
        """Check if recent SLAM measurements are available"""
        return self.ekf.is_slam_recent()
    
    def reset(
        self,
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None
    ):
        """
        Reset sensor fusion
        
        Args:
            position: New position (default: origin)
            orientation: New orientation (default: identity)
        """
        if position is None:
            position = np.zeros(3)
        if orientation is None:
            orientation = np.array([1, 0, 0, 0])
        
        new_state = EKFState(
            position=position,
            velocity=np.zeros(3),
            quaternion=orientation,
            gyro_bias=np.zeros(3),
            accel_bias=np.zeros(3),
            timestamp=time.time()
        )
        
        self.ekf.reset(new_state)
        
        with self.state_lock:
            self.fused_state = new_state.copy()
            self.fused_covariance = None
        
        # Clear queues
        while not self.imu_queue.empty():
            try:
                self.imu_queue.get_nowait()
            except Empty:
                break
        
        while not self.slam_queue.empty():
            try:
                self.slam_queue.get_nowait()
            except Empty:
                break
        
        print("[Sensor Fusion] Reset complete")
    
    def _processing_loop(self):
        """Main processing loop"""
        print("[Sensor Fusion] Processing loop started")
        
        dt_target = 1.0 / self.imu_rate_hz
        
        while self.running:
            loop_start = time.time()
            
            # Process all available SLAM updates first
            slam_updated = False
            while not self.slam_queue.empty():
                try:
                    slam = self.slam_queue.get_nowait()
                    self.ekf.update(slam)
                    self.slam_count += 1
                    slam_updated = True
                except Empty:
                    break
            
            # Process IMU updates
            imu_processed = False
            try:
                imu = self.imu_queue.get(timeout=0.01)
                self.ekf.predict(imu)
                self.imu_count += 1
                imu_processed = True
            except Empty:
                pass
            
            # Update fused state if anything was processed
            if imu_processed or slam_updated:
                with self.state_lock:
                    self.fused_state, self.fused_covariance = self.ekf.get_pose()
                
                # Log if enabled
                if self.log_file and self.fused_state:
                    self._log_state(slam_updated)
            
            # Rate limiting
            elapsed = time.time() - loop_start
            if elapsed < dt_target:
                time.sleep(dt_target - elapsed)
        
        print("[Sensor Fusion] Processing loop ended")
    
    def _log_state(self, slam_update: bool):
        """Log current state to file"""
        s = self.fused_state
        self.log_file.write(
            f"{s.timestamp},{s.position[0]},{s.position[1]},{s.position[2]},"
            f"{s.velocity[0]},{s.velocity[1]},{s.velocity[2]},"
            f"{s.quaternion[0]},{s.quaternion[1]},{s.quaternion[2]},{s.quaternion[3]},"
            f"{int(slam_update)}\n"
        )
    
    def _print_statistics(self):
        """Print processing statistics"""
        runtime = time.time() - self.start_time
        imu_rate = self.imu_count / runtime if runtime > 0 else 0
        slam_rate = self.slam_count / runtime if runtime > 0 else 0
        
        print("\n=== Sensor Fusion Statistics ===")
        print(f"Runtime: {runtime:.1f}s")
        print(f"IMU updates: {self.imu_count} ({imu_rate:.1f} Hz)")
        print(f"SLAM updates: {self.slam_count} ({slam_rate:.1f} Hz)")
        print(f"EKF stats: {self.ekf.get_statistics()}")


# Example usage
if __name__ == "__main__":
    # Initialize
    fusion = SensorFusion(
        initial_position=np.array([0, 0, 0]),
        initial_orientation=np.array([1, 0, 0, 0]),
        imu_rate_hz=200.0,
        enable_logging=True
    )
    
    fusion.start()
    
    try:
        # Simulate sensors
        for i in range(1000):
            t = i / 200.0  # 200 Hz
            
            # IMU at 200 Hz
            fusion.add_imu(
                accel=np.array([0, 0, 9.81]) + np.random.randn(3) * 0.1,
                gyro=np.array([0, 0, 0.1]) + np.random.randn(3) * 0.01,
                timestamp=t
            )
            
            # SLAM at 30 Hz
            if i % 7 == 0:
                fusion.add_slam(
                    position=np.array([t*0.1, 0, 0]),
                    quaternion=np.array([1, 0, 0, 0]),
                    timestamp=t,
                    tracking_ok=True,
                    confidence=0.9
                )
            
            # Print every second
            if i % 200 == 0:
                pos, quat, cov = fusion.get_pose()
                vel = fusion.get_velocity()
                print(f"\nTime: {t:.2f}s")
                print(f"Position: {pos}")
                print(f"Velocity: {vel}")
                print(f"SLAM available: {fusion.is_slam_available()}")
            
            time.sleep(0.005)  # 200 Hz
    
    except KeyboardInterrupt:
        print("\nInterrupted")
    
    finally:
        fusion.stop()