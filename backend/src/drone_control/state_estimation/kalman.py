"""
Extended Kalman Filter for IMU-SLAM Fusion
Fuses high-rate IMU (200-500 Hz) with low-rate SLAM (20-30 Hz)
Produces smooth, drift-corrected pose at high frequency
"""
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from scipy.spatial.transform import Rotation
from collections import deque
import time


@dataclass
class EKFState:
    """EKF state vector"""
    position: np.ndarray      # [x, y, z] meters
    velocity: np.ndarray      # [vx, vy, vz] m/s
    quaternion: np.ndarray    # [qw, qx, qy, qz]
    gyro_bias: np.ndarray     # [bx, by, bz] rad/s
    accel_bias: np.ndarray    # [bx, by, bz] m/s^2
    timestamp: float          # seconds
    
    def copy(self):
        """Create deep copy of state"""
        return EKFState(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            quaternion=self.quaternion.copy(),
            gyro_bias=self.gyro_bias.copy(),
            accel_bias=self.accel_bias.copy(),
            timestamp=self.timestamp
        )
    
    def to_vector(self) -> np.ndarray:
        """Convert state to 16D vector for EKF"""
        return np.concatenate([
            self.position,      # 3
            self.velocity,      # 3
            self.quaternion,    # 4
            self.gyro_bias,     # 3
            self.accel_bias     # 3
        ])  # Total: 16
    
    @staticmethod
    def from_vector(vec: np.ndarray, timestamp: float) -> 'EKFState':
        """Create state from 16D vector"""
        return EKFState(
            position=vec[0:3],
            velocity=vec[3:6],
            quaternion=vec[6:10],
            gyro_bias=vec[10:13],
            accel_bias=vec[13:16],
            timestamp=timestamp
        )


@dataclass
class IMUData:
    """IMU measurement"""
    accel: np.ndarray         # [ax, ay, az] m/s^2
    gyro: np.ndarray          # [gx, gy, gz] rad/s
    timestamp: float          # seconds


@dataclass
class SLAMPose:
    """SLAM pose measurement"""
    position: np.ndarray      # [x, y, z] meters
    quaternion: np.ndarray    # [qw, qx, qy, qz]
    timestamp: float          # seconds
    tracking_ok: bool         # True if SLAM tracking is good
    confidence: float = 1.0   # 0-1, higher = more confident


@dataclass
class EKFConfig:
    """EKF configuration parameters"""
    # Process noise (Q matrix diagonal values)
    position_process_noise: float = 0.01       # m^2
    velocity_process_noise: float = 0.1        # (m/s)^2
    quaternion_process_noise: float = 0.001    # rad^2
    gyro_bias_process_noise: float = 1e-6      # (rad/s)^2
    accel_bias_process_noise: float = 1e-4     # (m/s^2)^2
    
    # Measurement noise (R matrix diagonal values)
    slam_position_noise: float = 0.05          # m^2
    slam_quaternion_noise: float = 0.01        # rad^2
    
    # IMU noise characteristics
    gyro_noise: float = 1.7e-4                 # rad/s/sqrt(Hz)
    accel_noise: float = 2.0e-3                # m/s^2/sqrt(Hz)
    
    # Other parameters
    gravity: float = 9.81                      # m/s^2
    max_imu_dt: float = 0.1                    # Maximum time between IMU samples
    slam_timeout: float = 1.0                  # Consider SLAM lost after this time


class ExtendedKalmanFilter:
    """
    Extended Kalman Filter for IMU-SLAM sensor fusion
    
    State vector (16D):
        - Position (3D): x, y, z
        - Velocity (3D): vx, vy, vz  
        - Orientation (4D quaternion): qw, qx, qy, qz
        - Gyro bias (3D): bx, by, bz
        - Accel bias (3D): bx, by, bz
    """
    
    def __init__(self, initial_state: EKFState, config: Optional[EKFConfig] = None):
        """
        Initialize EKF
        
        Args:
            initial_state: Initial state estimate
            config: EKF configuration (uses defaults if None)
        """
        self.config = config or EKFConfig()
        
        # State
        self.state = initial_state.copy()
        
        # Covariance matrix (16x16)
        self.P = np.eye(16) * 0.1
        
        # Initialize process noise matrix Q
        self._initialize_Q()
        
        # Initialize measurement noise matrix R
        self._initialize_R()
        
        # Tracking
        self.last_imu_time = initial_state.timestamp
        self.last_slam_time = None
        self.prediction_count = 0
        self.update_count = 0
        
        # History for debugging
        self.state_history = deque(maxlen=1000)
        
        print("[EKF] Initialized")
        print(f"  Initial position: {initial_state.position}")
        print(f"  Initial orientation: {initial_state.quaternion}")
    
    def _initialize_Q(self):
        """Initialize process noise covariance matrix"""
        self.Q = np.diag([
            self.config.position_process_noise,
            self.config.position_process_noise,
            self.config.position_process_noise,
            self.config.velocity_process_noise,
            self.config.velocity_process_noise,
            self.config.velocity_process_noise,
            self.config.quaternion_process_noise,
            self.config.quaternion_process_noise,
            self.config.quaternion_process_noise,
            self.config.quaternion_process_noise,
            self.config.gyro_bias_process_noise,
            self.config.gyro_bias_process_noise,
            self.config.gyro_bias_process_noise,
            self.config.accel_bias_process_noise,
            self.config.accel_bias_process_noise,
            self.config.accel_bias_process_noise,
        ])
    
    def _initialize_R(self):
        """Initialize measurement noise covariance matrix"""
        self.R = np.diag([
            self.config.slam_position_noise,
            self.config.slam_position_noise,
            self.config.slam_position_noise,
            self.config.slam_quaternion_noise,
            self.config.slam_quaternion_noise,
            self.config.slam_quaternion_noise,
            self.config.slam_quaternion_noise,
        ])
    
    def predict(self, imu: IMUData) -> EKFState:
        """
        Prediction step using IMU measurements
        
        Args:
            imu: IMU measurement (accel, gyro, timestamp)
            
        Returns:
            Predicted state
        """
        # Compute time step
        dt = imu.timestamp - self.last_imu_time
        
        if dt <= 0 or dt > self.config.max_imu_dt:
            # Invalid dt, skip prediction
            return self.state.copy()
        
        # Remove biases from IMU measurements
        gyro_corrected = imu.gyro - self.state.gyro_bias
        accel_corrected = imu.accel - self.state.accel_bias
        
        # Predict state using IMU kinematics
        new_state = self._predict_state(
            self.state,
            accel_corrected,
            gyro_corrected,
            dt
        )
        
        # Compute Jacobian F (state transition matrix)
        F = self._compute_F(accel_corrected, gyro_corrected, dt)
        
        # Predict covariance: P = F * P * F^T + Q
        self.P = F @ self.P @ F.T + self.Q * dt
        
        # Update state
        self.state = new_state
        self.last_imu_time = imu.timestamp
        self.prediction_count += 1
        
        # Log state
        self.state_history.append(self.state.copy())
        
        return self.state.copy()
    
    def _predict_state(
        self,
        state: EKFState,
        accel: np.ndarray,
        gyro: np.ndarray,
        dt: float
    ) -> EKFState:
        """Predict next state using IMU kinematics"""
        # Convert quaternion to rotation matrix
        R = self._quat_to_rotation(state.quaternion)
        
        # Rotate acceleration to world frame and remove gravity
        accel_world = R @ accel - np.array([0, 0, self.config.gravity])
        
        # Update position and velocity
        new_position = state.position + state.velocity * dt + 0.5 * accel_world * dt**2
        new_velocity = state.velocity + accel_world * dt
        
        # Update orientation using gyroscope
        new_quaternion = self._integrate_quaternion(state.quaternion, gyro, dt)
        
        # Biases remain constant (random walk model)
        new_gyro_bias = state.gyro_bias.copy()
        new_accel_bias = state.accel_bias.copy()
        
        return EKFState(
            position=new_position,
            velocity=new_velocity,
            quaternion=new_quaternion,
            gyro_bias=new_gyro_bias,
            accel_bias=new_accel_bias,
            timestamp=state.timestamp + dt
        )
    
    def _integrate_quaternion(
        self,
        q: np.ndarray,
        omega: np.ndarray,
        dt: float
    ) -> np.ndarray:
        """
        Integrate quaternion using angular velocity
        Uses first-order integration
        """
        qw, qx, qy, qz = q
        wx, wy, wz = omega
        
        # Quaternion derivative
        q_dot = 0.5 * np.array([
            -qx*wx - qy*wy - qz*wz,
            qw*wx + qy*wz - qz*wy,
            qw*wy - qx*wz + qz*wx,
            qw*wz + qx*wy - qy*wx
        ])
        
        # Integrate
        q_new = q + q_dot * dt
        
        # Normalize
        return q_new / np.linalg.norm(q_new)
    
    def _compute_F(
        self,
        accel: np.ndarray,
        gyro: np.ndarray,
        dt: float
    ) -> np.ndarray:
        """Compute state transition Jacobian F"""
        F = np.eye(16)
        
        # Get rotation matrix
        R = self._quat_to_rotation(self.state.quaternion)
        
        # Position depends on velocity
        F[0:3, 3:6] = np.eye(3) * dt
        
        # Velocity depends on orientation and accel bias
        F[3:6, 6:10] = self._compute_velocity_orientation_jacobian(accel, dt)
        F[3:6, 13:16] = -R * dt
        
        # Orientation depends on gyro and gyro bias
        F[6:10, 6:10] = self._compute_quaternion_jacobian(gyro, dt)
        F[6:10, 10:13] = self._compute_quaternion_gyro_bias_jacobian(dt)
        
        return F
    
    def _compute_velocity_orientation_jacobian(
        self,
        accel: np.ndarray,
        dt: float
    ) -> np.ndarray:
        """Compute jacobian of velocity w.r.t. quaternion"""
        # Simplified: derivative of R*a w.r.t. quaternion
        # This is a 3x4 matrix
        R = self._quat_to_rotation(self.state.quaternion)
        skew_a = self._skew_symmetric(accel)
        return -R @ skew_a * dt
    
    def _compute_quaternion_jacobian(
        self,
        omega: np.ndarray,
        dt: float
    ) -> np.ndarray:
        """Compute quaternion update Jacobian"""
        wx, wy, wz = omega
        
        Omega = np.array([
            [0, -wx, -wy, -wz],
            [wx, 0, wz, -wy],
            [wy, -wz, 0, wx],
            [wz, wy, -wx, 0]
        ])
        
        return np.eye(4) + 0.5 * Omega * dt
    
    def _compute_quaternion_gyro_bias_jacobian(self, dt: float) -> np.ndarray:
        """Compute jacobian of quaternion w.r.t. gyro bias"""
        qw, qx, qy, qz = self.state.quaternion
        
        return -0.5 * dt * np.array([
            [qx, qy, qz],
            [-qw, qz, -qy],
            [-qz, -qw, qx],
            [qy, -qx, -qw]
        ])
    
    def update(self, slam_pose: SLAMPose) -> EKFState:
        """
        Update step using SLAM measurement
        
        Args:
            slam_pose: SLAM pose measurement
            
        Returns:
            Updated state
        """
        if not slam_pose.tracking_ok:
            # SLAM lost, skip update
            print("[EKF] SLAM tracking lost, skipping update")
            return self.state.copy()
        
        # Measurement vector: [position, quaternion]
        z = np.concatenate([slam_pose.position, slam_pose.quaternion])
        
        # Predicted measurement
        h = np.concatenate([self.state.position, self.state.quaternion])
        
        # Measurement Jacobian H (7x16)
        H = np.zeros((7, 16))
        H[0:3, 0:3] = np.eye(3)  # Position measurement
        H[3:7, 6:10] = np.eye(4)  # Quaternion measurement
        
        # Adjust R based on confidence
        R_adjusted = self.R * (2.0 - slam_pose.confidence)
        
        # Innovation
        y = z - h
        
        # Handle quaternion ambiguity (q and -q represent same rotation)
        if np.dot(slam_pose.quaternion, self.state.quaternion) < 0:
            y[3:7] = -slam_pose.quaternion - self.state.quaternion
        
        # Innovation covariance
        S = H @ self.P @ H.T + R_adjusted
        
        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Update state
        state_correction = K @ y
        new_state_vec = self.state.to_vector() + state_correction
        
        # Normalize quaternion
        new_state_vec[6:10] /= np.linalg.norm(new_state_vec[6:10])
        
        self.state = EKFState.from_vector(new_state_vec, slam_pose.timestamp)
        
        # Update covariance
        self.P = (np.eye(16) - K @ H) @ self.P
        
        self.last_slam_time = slam_pose.timestamp
        self.update_count += 1
        
        return self.state.copy()
    
    def get_pose(self) -> Tuple[EKFState, np.ndarray]:
        """
        Get current fused pose estimate
        
        Returns:
            (state, covariance): Current state and 16x16 covariance matrix
        """
        return self.state.copy(), self.P.copy()
    
    def get_position(self) -> np.ndarray:
        """Get current position [x, y, z]"""
        return self.state.position.copy()
    
    def get_velocity(self) -> np.ndarray:
        """Get current velocity [vx, vy, vz]"""
        return self.state.velocity.copy()
    
    def get_orientation(self) -> np.ndarray:
        """Get current orientation quaternion [qw, qx, qy, qz]"""
        return self.state.quaternion.copy()
    
    def get_orientation_rates(self) -> np.ndarray:
        """
        Get orientation rates [roll_rate, pitch_rate, yaw_rate]
        Estimated from recent quaternion changes
        """
        if len(self.state_history) < 2:
            return np.zeros(3)
        
        # Get last two states
        state_current = self.state_history[-1]
        state_prev = self.state_history[-2]
        
        dt = state_current.timestamp - state_prev.timestamp
        if dt <= 0:
            return np.zeros(3)
        
        # Compute relative rotation
        q_rel = self._quaternion_multiply(
            self._quaternion_conjugate(state_prev.quaternion),
            state_current.quaternion
        )
        
        # Convert to angular velocity
        angle = 2 * np.arccos(np.clip(q_rel[0], -1, 1))
        if np.abs(angle) < 1e-6:
            return np.zeros(3)
        
        axis = q_rel[1:4] / np.sin(angle / 2)
        omega = axis * angle / dt
        
        # Convert to roll, pitch, yaw rates (body frame)
        return omega
    
    def is_slam_recent(self) -> bool:
        """Check if SLAM measurements are recent"""
        if self.last_slam_time is None:
            return False
        return (self.state.timestamp - self.last_slam_time) < self.config.slam_timeout
    
    def reset(self, initial_state: Optional[EKFState] = None):
        """
        Reset EKF to initial state
        
        Args:
            initial_state: New initial state (if None, resets to zero state)
        """
        if initial_state is None:
            initial_state = EKFState(
                position=np.zeros(3),
                velocity=np.zeros(3),
                quaternion=np.array([1, 0, 0, 0]),
                gyro_bias=np.zeros(3),
                accel_bias=np.zeros(3),
                timestamp=time.time()
            )
        
        self.state = initial_state.copy()
        self.P = np.eye(16) * 0.1
        self.last_imu_time = initial_state.timestamp
        self.last_slam_time = None
        self.prediction_count = 0
        self.update_count = 0
        self.state_history.clear()
        
        print("[EKF] Reset complete")
    
    def get_statistics(self) -> dict:
        """Get filter statistics"""
        return {
            'predictions': self.prediction_count,
            'updates': self.update_count,
            'slam_recent': self.is_slam_recent(),
            'position_std': np.sqrt(np.diag(self.P)[0:3]),
            'velocity_std': np.sqrt(np.diag(self.P)[3:6]),
            'orientation_std': np.sqrt(np.diag(self.P)[6:10]),
        }
    
    # Helper functions
    @staticmethod
    def _quat_to_rotation(q: np.ndarray) -> np.ndarray:
        """Convert quaternion to rotation matrix"""
        return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    
    @staticmethod
    def _rotation_to_quat(R: np.ndarray) -> np.ndarray:
        """Convert rotation matrix to quaternion [qw, qx, qy, qz]"""
        q_scipy = Rotation.from_matrix(R).as_quat()
        return np.array([q_scipy[3], q_scipy[0], q_scipy[1], q_scipy[2]])
    
    @staticmethod
    def _skew_symmetric(v: np.ndarray) -> np.ndarray:
        """Create skew-symmetric matrix from vector"""
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
    
    @staticmethod
    def _quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Multiply two quaternions"""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
    
    @staticmethod
    def _quaternion_conjugate(q: np.ndarray) -> np.ndarray:
        """Compute quaternion conjugate"""
        return np.array([q[0], -q[1], -q[2], -q[3]])


# Example usage
if __name__ == "__main__":
    # Initialize EKF
    initial_state = EKFState(
        position=np.array([0.0, 0.0, 0.0]),
        velocity=np.array([0.0, 0.0, 0.0]),
        quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
        gyro_bias=np.zeros(3),
        accel_bias=np.zeros(3),
        timestamp=0.0
    )
    
    ekf = ExtendedKalmanFilter(initial_state)
    
    # Simulate IMU updates at 200 Hz
    dt_imu = 1.0 / 200.0
    
    for i in range(1000):
        t = i * dt_imu
        
        # Simulate IMU
        imu = IMUData(
            accel=np.array([0.0, 0.0, 9.81]) + np.random.randn(3) * 0.1,
            gyro=np.array([0.0, 0.0, 0.1]) + np.random.randn(3) * 0.01,
            timestamp=t
        )
        
        # Predict
        ekf.predict(imu)
        
        # SLAM update every 30 frames (~30 Hz)
        if i % 30 == 0:
            slam = SLAMPose(
                position=np.array([t*0.1, 0.0, 0.0]),
                quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
                timestamp=t,
                tracking_ok=True,
                confidence=0.9
            )
            ekf.update(slam)
        
        # Print every second
        if i % 200 == 0:
            state, cov = ekf.get_pose()
            print(f"\nTime: {t:.2f}s")
            print(f"Position: {state.position}")
            print(f"Velocity: {state.velocity}")
            print(f"Stats: {ekf.get_statistics()}")