"""
Real-time Sensor Fusion Visualization
Plots IMU, SLAM, and fused estimates
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import time


class FusionVisualizer:
    """
    Real-time visualization of sensor fusion
    Shows position, velocity, and covariance
    """
    
    def __init__(self, history_length: int = 500):
        """
        Initialize visualizer
        
        Args:
            history_length: Number of samples to display
        """
        self.history_length = history_length
        
        # Data buffers
        self.time_history = deque(maxlen=history_length)
        
        # Position
        self.pos_fused = deque(maxlen=history_length)
        self.pos_slam = deque(maxlen=history_length)
        
        # Velocity
        self.vel_fused = deque(maxlen=history_length)
        
        # Uncertainty
        self.pos_std = deque(maxlen=history_length)
        
        # Setup plot
        self.fig = plt.figure(figsize=(15, 10))
        self._setup_plots()
        
        self.start_time = time.time()
        
    def _setup_plots(self):
        """Setup matplotlib subplots"""
        # 3D trajectory
        self.ax_3d = self.fig.add_subplot(2, 3, 1, projection='3d')
        self.ax_3d.set_xlabel('X (m)')
        self.ax_3d.set_ylabel('Y (m)')
        self.ax_3d.set_zlabel('Z (m)')
        self.ax_3d.set_title('3D Trajectory')
        
        # Position X
        self.ax_px = self.fig.add_subplot(2, 3, 2)
        self.ax_px.set_ylabel('X Position (m)')
        self.ax_px.set_title('Position X')
        self.ax_px.grid(True)
        
        # Position Y
        self.ax_py = self.fig.add_subplot(2, 3, 3)
        self.ax_py.set_ylabel('Y Position (m)')
        self.ax_py.set_title('Position Y')
        self.ax_py.grid(True)
        
        # Position Z
        self.ax_pz = self.fig.add_subplot(2, 3, 4)
        self.ax_pz.set_ylabel('Z Position (m)')
        self.ax_pz.set_xlabel('Time (s)')
        self.ax_pz.set_title('Position Z')
        self.ax_pz.grid(True)
        
        # Velocity
        self.ax_vel = self.fig.add_subplot(2, 3, 5)
        self.ax_vel.set_ylabel('Velocity (m/s)')
        self.ax_vel.set_xlabel('Time (s)')
        self.ax_vel.set_title('Velocity')
        self.ax_vel.grid(True)
        
        # Uncertainty
        self.ax_unc = self.fig.add_subplot(2, 3, 6)
        self.ax_unc.set_ylabel('Position Std (m)')
        self.ax_unc.set_xlabel('Time (s)')
        self.ax_unc.set_title('Position Uncertainty')
        self.ax_unc.grid(True)
        
        plt.tight_layout()
    
    def update(
        self,
        fused_pos: np.ndarray,
        fused_vel: np.ndarray,
        covariance: np.ndarray,
        slam_pos: np.ndarray = None,
        timestamp: float = None
    ):
        """
        Update visualization with new data
        
        Args:
            fused_pos: Fused position [x, y, z]
            fused_vel: Fused velocity [vx, vy, vz]
            covariance: State covariance matrix
            slam_pos: SLAM position (optional)
            timestamp: Current time (uses internal if None)
        """
        if timestamp is None:
            timestamp = time.time() - self.start_time
        
        # Store data
        self.time_history.append(timestamp)
        self.pos_fused.append(fused_pos.copy())
        self.vel_fused.append(fused_vel.copy())
        
        if slam_pos is not None:
            self.pos_slam.append(slam_pos.copy())
        
        # Extract position std from covariance
        pos_std_val = np.sqrt(np.diag(covariance)[0:3])
        self.pos_std.append(np.linalg.norm(pos_std_val))
    
    def render(self):
        """Render current visualization"""
        if len(self.pos_fused) < 2:
            return
        
        # Convert to arrays
        times = np.array(self.time_history)
        pos_fused_arr = np.array(self.pos_fused)
        vel_fused_arr = np.array(self.vel_fused)
        pos_std_arr = np.array(self.pos_std)
        
        # Clear all axes
        self.ax_3d.cla()
        self.ax_px.cla()
        self.ax_py.cla()
        self.ax_pz.cla()
        self.ax_vel.cla()
        self.ax_unc.cla()
        
        # 3D trajectory
        self.ax_3d.plot(pos_fused_arr[:, 0], pos_fused_arr[:, 1], pos_fused_arr[:, 2],
                       'b-', label='Fused', linewidth=2)
        if len(self.pos_slam) > 0:
            pos_slam_arr = np.array(self.pos_slam)
            self.ax_3d.scatter(pos_slam_arr[:, 0], pos_slam_arr[:, 1], pos_slam_arr[:, 2],
                              c='r', marker='o', s=20, label='SLAM', alpha=0.5)
        
        # Current position
        self.ax_3d.scatter([pos_fused_arr[-1, 0]], [pos_fused_arr[-1, 1]], [pos_fused_arr[-1, 2]],
                          c='g', marker='o', s=100, label='Current')
        
        self.ax_3d.set_xlabel('X (m)')
        self.ax_3d.set_ylabel('Y (m)')
        self.ax_3d.set_zlabel('Z (m)')
        self.ax_3d.set_title('3D Trajectory')
        self.ax_3d.legend()
        
        # Position X
        self.ax_px.plot(times, pos_fused_arr[:, 0], 'b-', label='Fused', linewidth=2)
        if len(self.pos_slam) > 0:
            slam_times = times[-len(self.pos_slam):]
            self.ax_px.scatter(slam_times, pos_slam_arr[:, 0], c='r', s=20, label='SLAM', alpha=0.5)
        self.ax_px.set_ylabel('X (m)')
        self.ax_px.set_title('Position X')
        self.ax_px.legend()
        self.ax_px.grid(True)
        
        # Position Y
        self.ax_py.plot(times, pos_fused_arr[:, 1], 'b-', label='Fused', linewidth=2)
        if len(self.pos_slam) > 0:
            self.ax_py.scatter(slam_times, pos_slam_arr[:, 1], c='r', s=20, label='SLAM', alpha=0.5)
        self.ax_py.set_ylabel('Y (m)')
        self.ax_py.set_title('Position Y')
        self.ax_py.legend()
        self.ax_py.grid(True)
        
        # Position Z
        self.ax_pz.plot(times, pos_fused_arr[:, 2], 'b-', label='Fused', linewidth=2)
        if len(self.pos_slam) > 0:
            self.ax_pz.scatter(slam_times, pos_slam_arr[:, 2], c='r', s=20, label='SLAM', alpha=0.5)
        self.ax_pz.set_ylabel('Z (m)')
        self.ax_pz.set_xlabel('Time (s)')
        self.ax_pz.set_title('Position Z')
        self.ax_pz.legend()
        self.ax_pz.grid(True)
        
        # Velocity magnitude
        vel_mag = np.linalg.norm(vel_fused_arr, axis=1)
        self.ax_vel.plot(times, vel_mag, 'g-', linewidth=2)
        self.ax_vel.set_ylabel('Speed (m/s)')
        self.ax_vel.set_xlabel('Time (s)')
        self.ax_vel.set_title('Velocity Magnitude')
        self.ax_vel.grid(True)
        
        # Uncertainty
        self.ax_unc.plot(times, pos_std_arr, 'orange', linewidth=2)
        self.ax_unc.set_ylabel('Position Std (m)')
        self.ax_unc.set_xlabel('Time (s)')
        self.ax_unc.set_title('Position Uncertainty')
        self.ax_unc.grid(True)
        
        plt.tight_layout()
        plt.pause(0.001)
    
    def show(self):
        """Show the plot"""
        plt.show()


# Example usage with sensor fusion
if __name__ == "__main__":
    from sensor_fusion import SensorFusion
    import time
    
    # Initialize
    fusion = SensorFusion(enable_logging=False)
    fusion.start()
    
    viz = FusionVisualizer(history_length=300)
    
    plt.ion()
    
    try:
        # Simulate flight
        for i in range(1000):
            t = i / 200.0
            
            # Add IMU (200 Hz)
            accel = np.array([0, 0, 9.81]) + np.random.randn(3) * 0.1
            gyro = np.array([0, 0, 0.2]) + np.random.randn(3) * 0.01
            fusion.add_imu(accel, gyro, t)
            
            # Add SLAM (30 Hz)
            slam_pos = None
            if i % 7 == 0:
                # Circular motion
                slam_pos = np.array([
                    np.cos(t) * 2.0,
                    np.sin(t) * 2.0,
                    0.5 + np.sin(t * 2) * 0.2
                ])
                fusion.add_slam(
                    position=slam_pos,
                    quaternion=np.array([1, 0, 0, 0]),
                    timestamp=t,
                    tracking_ok=True,
                    confidence=0.9
                )
            
            # Update visualization (10 Hz)
            if i % 20 == 0:
                pos, quat, cov = fusion.get_pose()
                vel = fusion.get_velocity()
                
                viz.update(
                    fused_pos=pos,
                    fused_vel=vel,
                    covariance=cov,
                    slam_pos=slam_pos,
                    timestamp=t
                )
                viz.render()
            
            time.sleep(0.005)
    
    except KeyboardInterrupt:
        print("\nStopped")
    
    finally:
        fusion.stop()
        plt.ioff()
        viz.show()