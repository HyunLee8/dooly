"""
Error-State Kalman Filter (ESKF) for IMU-SLAM Sensor Fusion
Fuses high-rate IMU data with lower-rate SLAM poses for drone state estimation
Uses error-state formulation for improved numerical stability and quaternion handling
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
import time


@dataclass
class IMUData:
    """IMU measurement data"""
    ax: float  # m/s^2
    ay: float
    az: float
    gx: float  # rad/s
    gy: float
    gz: float
    timestamp: float


@dataclass
class SLAMPose:
    """SLAM pose measurement"""
    x: float
    y: float
    z: float
    qx: float  # quaternion
    qy: float
    qz: float
    qw: float
    tracking_status: bool
    timestamp: float


@dataclass
class FusedPose:
    """Output fused pose"""
    x: float
    y: float
    z: float
    qx: float
    qy: float
    qz: float
    qw: float
    vx: float
    vy: float
    vz: float
    covariance: np.ndarray
    timestamp: float


class IMUSLAMFusion:
    """Error-State Kalman Filter for fusing IMU and SLAM data"""
    
    def __init__(self, 
                 process_noise_acc: float = 0.1,
                 process_noise_gyro: float = 0.01,
                 process_noise_bias: float = 0.001,
                 measurement_noise_pos: float = 0.05,
                 measurement_noise_orient: float = 0.02):
        """
        Initialize ESKF with configurable noise parameters
        
        Args:
            process_noise_acc: Accelerometer process noise std dev (m/s^2)
            process_noise_gyro: Gyroscope process noise std dev (rad/s)
            process_noise_bias: Bias drift noise std dev
            measurement_noise_pos: SLAM position measurement noise std dev (m)
            measurement_noise_orient: SLAM orientation measurement noise std dev (rad)
        """
        # Nominal state: [p, q, v, ba, bg]
        # Position (3) + Quaternion (4) + Velocity (3) + Accel bias (3) + Gyro bias (3) = 16
        self.nominal_state = np.zeros(16)
        self.nominal_state[3] = 1.0  # Initialize quaternion to identity (qw=1)
        
        # Error state: [δp, δθ, δv, δba, δbg]
        # Position error (3) + Rotation error (3) + Velocity error (3) + Bias errors (6) = 15
        # Note: rotation error is 3D (axis-angle), not 4D quaternion
        self.error_state = np.zeros(15)
        
        # Error state covariance matrix (15x15)
        self.P = np.eye(15) * 0.1
        
        # Process noise covariance (continuous time)
        # Applied to: [na, ng, nba, nbg] = 12 noise terms
        self.Qi = np.diag([
            process_noise_acc**2, process_noise_acc**2, process_noise_acc**2,  # accel noise
            process_noise_gyro**2, process_noise_gyro**2, process_noise_gyro**2,  # gyro noise
            process_noise_bias**2, process_noise_bias**2, process_noise_bias**2,  # accel bias drift
            process_noise_bias**2, process_noise_bias**2, process_noise_bias**2   # gyro bias drift
        ])
        
        # Measurement noise covariance R (for SLAM updates)
        self.R = np.diag([
            measurement_noise_pos**2, measurement_noise_pos**2, measurement_noise_pos**2,  # position
            measurement_noise_orient**2, measurement_noise_orient**2, measurement_noise_orient**2  # rotation
        ])
        
        self.last_predict_time = None
        self.gravity = np.array([0, 0, 9.81])  # Gravity vector in world frame (positive down for NED)
        
    def predict(self, imu_data: IMUData) -> FusedPose:
        """
        ESKF prediction step using IMU measurements
        
        Args:
            imu_data: IMU accelerometer and gyroscope data
            
        Returns:
            Predicted fused pose
        """
        # Calculate dt
        if self.last_predict_time is None:
            self.last_predict_time = imu_data.timestamp
            return self.get_pose()
        
        dt = imu_data.timestamp - self.last_predict_time
        if dt <= 0 or dt > 1.0:  # Sanity check
            dt = 0.01  # Default to 100Hz if timestamp issue
        self.last_predict_time = imu_data.timestamp
        
        # Extract nominal state components
        p = self.nominal_state[0:3]
        q = self.nominal_state[3:7]
        v = self.nominal_state[7:10]
        ba = self.nominal_state[10:13]
        bg = self.nominal_state[13:16]
        
        # IMU measurements
        am = np.array([imu_data.ax, imu_data.ay, imu_data.az])
        wm = np.array([imu_data.gx, imu_data.gy, imu_data.gz])
        
        # Remove biases from measurements
        a_unbiased = am - ba
        w_unbiased = wm - bg
        
        # Propagate nominal state
        R = self._quaternion_to_rotation_matrix(q)
        
        # Acceleration in world frame (subtract gravity in world frame)
        a_world = R @ a_unbiased - self.gravity
        
        # Update nominal state using simple integration
        p_new = p + v * dt + 0.5 * a_world * dt**2
        v_new = v + a_world * dt
        q_new = self._integrate_quaternion(q, w_unbiased, dt)
        ba_new = ba  # Biases evolve via random walk
        bg_new = bg
        
        # Update nominal state
        self.nominal_state[0:3] = p_new
        self.nominal_state[3:7] = q_new
        self.nominal_state[7:10] = v_new
        self.nominal_state[10:13] = ba_new
        self.nominal_state[13:16] = bg_new
        
        # Propagate error state covariance
        # Compute error state Jacobian Fx
        Fx = self._compute_error_state_jacobian(dt, a_unbiased, w_unbiased, R)
        
        # Compute noise Jacobian Fi
        Fi = self._compute_noise_jacobian(dt, R)
        
        # Propagate covariance: P = Fx * P * Fx^T + Fi * Qi * Fi^T
        self.P = Fx @ self.P @ Fx.T + Fi @ self.Qi @ Fi.T
        
        # Error state remains zero after prediction (we integrate it into nominal state)
        self.error_state = np.zeros(15)
        
        return self.get_pose()
    
    def update(self, slam_pose: SLAMPose, slam_status: bool = True) -> FusedPose:
        """
        ESKF correction step using SLAM pose measurement
        
        Args:
            slam_pose: SLAM position and orientation measurement
            slam_status: Whether SLAM tracking is valid
            
        Returns:
            Updated fused pose
        """
        if not slam_status:
            # SLAM tracking lost, skip update
            return self.get_pose()
        
        # Extract nominal state
        p_nominal = self.nominal_state[0:3]
        q_nominal = self.nominal_state[3:7]
        
        # SLAM measurement
        p_meas = np.array([slam_pose.x, slam_pose.y, slam_pose.z])
        q_meas = np.array([slam_pose.qw, slam_pose.qx, slam_pose.qy, slam_pose.qz])
        q_meas = self._normalize_quaternion(q_meas)
        
        # Handle quaternion sign ambiguity
        if np.dot(q_nominal, q_meas) < 0:
            q_meas = -q_meas
        
        # Innovation (measurement residual)
        # Position innovation
        y_p = p_meas - p_nominal
        
        # Rotation innovation (convert quaternion difference to axis-angle)
        q_error = self._quaternion_multiply(self._quaternion_inverse(q_nominal), q_meas)
        y_theta = self._quaternion_to_rotation_vector(q_error)
        
        # Combined innovation
        y = np.concatenate([y_p, y_theta])
        
        # Measurement Jacobian H
        # Measurement model: z = [p, θ] = h(x)
        # For error state: δz = H * δx
        H = np.zeros((6, 15))
        H[0:3, 0:3] = np.eye(3)  # Position error
        H[3:6, 3:6] = np.eye(3)  # Rotation error
        
        # Innovation covariance
        S = H @ self.P @ H.T + self.R
        
        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Update error state
        self.error_state = K @ y
        
        # Inject error state into nominal state
        self._inject_error_state()
        
        # Update covariance (Joseph form for numerical stability)
        I_KH = np.eye(15) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        
        # Reset error state after injection
        self.error_state = np.zeros(15)
        
        return self.get_pose()
    
    def get_pose(self) -> FusedPose:
        """
        Get current fused pose estimate
        
        Returns:
            Current fused pose with position, orientation, velocity, and covariance
        """
        return FusedPose(
            x=self.nominal_state[0],
            y=self.nominal_state[1],
            z=self.nominal_state[2],
            qw=self.nominal_state[3],
            qx=self.nominal_state[4],
            qy=self.nominal_state[5],
            qz=self.nominal_state[6],
            vx=self.nominal_state[7],
            vy=self.nominal_state[8],
            vz=self.nominal_state[9],
            covariance=self.P.copy(),
            timestamp=time.time()
        )
    
    def reset(self, initial_pose: Optional[SLAMPose] = None):
        """
        Reset ESKF state
        
        Args:
            initial_pose: Optional initial pose to set, otherwise reset to zero
        """
        self.nominal_state = np.zeros(16)
        self.nominal_state[3] = 1.0  # Identity quaternion
        
        if initial_pose is not None:
            self.nominal_state[0:3] = [initial_pose.x, initial_pose.y, initial_pose.z]
            self.nominal_state[3:7] = [initial_pose.qw, initial_pose.qx, initial_pose.qy, initial_pose.qz]
            self.nominal_state[3:7] = self._normalize_quaternion(self.nominal_state[3:7])
        
        self.error_state = np.zeros(15)
        self.P = np.eye(15) * 0.1
        self.last_predict_time = None
    
    # ========== Helper Methods ==========
    
    def _inject_error_state(self):
        """Inject error state into nominal state and reset error state"""
        # Position injection
        self.nominal_state[0:3] += self.error_state[0:3]
        
        # Rotation injection (axis-angle to quaternion)
        delta_theta = self.error_state[3:6]
        delta_q = self._rotation_vector_to_quaternion(delta_theta)
        q_nominal = self.nominal_state[3:7]
        self.nominal_state[3:7] = self._quaternion_multiply(q_nominal, delta_q)
        self.nominal_state[3:7] = self._normalize_quaternion(self.nominal_state[3:7])
        
        # Velocity injection
        self.nominal_state[7:10] += self.error_state[6:9]
        
        # Bias injection
        self.nominal_state[10:13] += self.error_state[9:12]
        self.nominal_state[13:16] += self.error_state[12:15]
    
    def _compute_error_state_jacobian(self, dt: float, a: np.ndarray, 
                                      w: np.ndarray, R: np.ndarray) -> np.ndarray:
        """Compute Jacobian of error state transition"""
        Fx = np.eye(15)
        
        # Position error dynamics: δp_dot = δv
        Fx[0:3, 6:9] = np.eye(3) * dt
        
        # Rotation error dynamics: δθ_dot = -[w]_x * δθ - δbg
        Fx[3:6, 3:6] = np.eye(3) - self._skew_symmetric(w) * dt
        Fx[3:6, 12:15] = -np.eye(3) * dt
        
        # Velocity error dynamics: δv_dot = -R * [a]_x * δθ - R * δba
        Fx[6:9, 3:6] = -R @ self._skew_symmetric(a) * dt
        Fx[6:9, 9:12] = -R * dt
        
        # Biases are random walk (identity already in Fx)
        
        return Fx
    
    def _compute_noise_jacobian(self, dt: float, R: np.ndarray) -> np.ndarray:
        """Compute Jacobian mapping noise to error state"""
        Fi = np.zeros((15, 12))
        
        # Acceleration noise affects velocity
        Fi[6:9, 0:3] = -R * dt
        
        # Gyroscope noise affects rotation
        Fi[3:6, 3:6] = -np.eye(3) * dt
        
        # Bias noise affects biases directly
        Fi[9:12, 6:9] = np.eye(3) * dt
        Fi[12:15, 9:12] = np.eye(3) * dt
        
        return Fi
    
    def _normalize_quaternion(self, q: np.ndarray) -> np.ndarray:
        """Normalize quaternion to unit length"""
        norm = np.linalg.norm(q)
        if norm > 1e-9:
            return q / norm
        return np.array([1.0, 0.0, 0.0, 0.0])
    
    def _quaternion_to_rotation_matrix(self, q: np.ndarray) -> np.ndarray:
        """Convert quaternion to rotation matrix"""
        qw, qx, qy, qz = q
        
        R = np.array([
            [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
            [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
            [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]
        ])
        
        return R
    
    def _integrate_quaternion(self, q: np.ndarray, omega: np.ndarray, dt: float) -> np.ndarray:
        """Integrate quaternion using angular velocity (first-order)"""
        qw, qx, qy, qz = q
        wx, wy, wz = omega
        
        # Quaternion derivative matrix
        Omega = np.array([
            [0, -wx, -wy, -wz],
            [wx, 0, wz, -wy],
            [wy, -wz, 0, wx],
            [wz, wy, -wx, 0]
        ])
        
        # First-order integration: q_new = q + 0.5 * Omega * q * dt
        q_new = q + 0.5 * Omega @ q * dt
        
        return self._normalize_quaternion(q_new)
    
    def _quaternion_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Multiply two quaternions: q1 * q2"""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
    
    def _quaternion_inverse(self, q: np.ndarray) -> np.ndarray:
        """Compute quaternion inverse (conjugate for unit quaternions)"""
        return np.array([q[0], -q[1], -q[2], -q[3]])
    
    def _quaternion_to_rotation_vector(self, q: np.ndarray) -> np.ndarray:
        """Convert small rotation quaternion to rotation vector (axis-angle)"""
        qw, qx, qy, qz = q
        
        # For small rotations: θ ≈ 2 * [qx, qy, qz]
        # For larger rotations use proper conversion
        angle = 2 * np.arccos(np.clip(qw, -1.0, 1.0))
        
        if angle < 1e-6:
            return 2 * np.array([qx, qy, qz])
        
        sin_half = np.sin(angle / 2)
        if abs(sin_half) < 1e-9:
            return np.zeros(3)
        
        axis = np.array([qx, qy, qz]) / sin_half
        return angle * axis
    
    def _rotation_vector_to_quaternion(self, rv: np.ndarray) -> np.ndarray:
        """Convert rotation vector (axis-angle) to quaternion"""
        angle = np.linalg.norm(rv)
        
        if angle < 1e-6:
            # Small angle approximation
            return np.array([1.0, rv[0]/2, rv[1]/2, rv[2]/2])
        
        axis = rv / angle
        half_angle = angle / 2
        
        return np.array([
            np.cos(half_angle),
            axis[0] * np.sin(half_angle),
            axis[1] * np.sin(half_angle),
            axis[2] * np.sin(half_angle)
        ])
    
    def _skew_symmetric(self, v: np.ndarray) -> np.ndarray:
        """Create skew-symmetric matrix from vector"""
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])


# ========== Example Usage ==========
if __name__ == "__main__":
    # Initialize filter
    kf = IMUSLAMFusion(
        process_noise_acc=0.1,
        process_noise_gyro=0.01,
        measurement_noise_pos=0.05,
        measurement_noise_orient=0.02
    )
    
    # Simulate IMU data (high rate - e.g., 100Hz)
    imu = IMUData(
        ax=0.1, ay=0.05, az=9.81,
        gx=0.01, gy=0.02, gz=0.0,
        timestamp=time.time()
    )
    
    # Prediction step
    predicted_pose = kf.predict(imu)
    print(f"Predicted pose: x={predicted_pose.x:.3f}, y={predicted_pose.y:.3f}, z={predicted_pose.z:.3f}")
    
    # Simulate SLAM pose (low rate - e.g., 10Hz)
    slam = SLAMPose(
        x=0.5, y=0.3, z=1.2,
        qw=1.0, qx=0.0, qy=0.0, qz=0.0,
        tracking_status=True,
        timestamp=time.time()
    )
    
    # Update step
    updated_pose = kf.update(slam, slam_status=True)
    print(f"Updated pose: x={updated_pose.x:.3f}, y={updated_pose.y:.3f}, z={updated_pose.z:.3f}")
    print(f"Velocity: vx={updated_pose.vx:.3f}, vy={updated_pose.vy:.3f}, vz={updated_pose.vz:.3f}")