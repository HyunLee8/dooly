"""
Tello Interface - Low-level wrapper for DJI Tello drone
Handles communication, video streaming, and state management
"""

import cv2
import numpy as np
import time
from typing import Optional, Callable
from dataclasses import dataclass
from threading import Thread, Lock
from djitellopy import Tello


@dataclass
class TelloState:
    """Tello drone state"""
    # Position and orientation
    x: float = 0.0  # cm
    y: float = 0.0  # cm
    z: float = 0.0  # cm
    pitch: float = 0.0  # degrees
    roll: float = 0.0  # degrees
    yaw: float = 0.0  # degrees
    
    # Velocities
    vx: float = 0.0  # cm/s
    vy: float = 0.0  # cm/s
    vz: float = 0.0  # cm/s
    
    # Accelerations
    agx: float = 0.0  # cm/s^2
    agy: float = 0.0  # cm/s^2
    agz: float = 0.0  # cm/s^2
    
    # Status
    battery: int = 0  # percentage
    temperature: int = 0  # celsius
    height: int = 0  # cm
    time_of_flight: int = 0  # cm
    barometer: float = 0.0  # cm
    
    # Flight state
    is_flying: bool = False
    
    # Timestamp
    timestamp: float = 0.0
    
    def get_position_m(self) -> np.ndarray:
        """Get position in meters"""
        return np.array([self.x, self.y, self.z]) / 100.0
    
    def get_velocity_mps(self) -> np.ndarray:
        """Get velocity in m/s"""
        return np.array([self.vx, self.vy, self.vz]) / 100.0
    
    def get_acceleration_mps2(self) -> np.ndarray:
        """Get acceleration in m/s^2"""
        return np.array([self.agx, self.agy, self.agz]) / 100.0
    
    def get_orientation_rad(self) -> np.ndarray:
        """Get orientation in radians"""
        return np.deg2rad([self.roll, self.pitch, self.yaw])


class TelloInterface:
    """
    Low-level interface to Tello drone
    Wraps djitellopy with additional functionality
    """
    
    def __init__(self):
        """Initialize Tello interface"""
        self.tello: Optional[Tello] = None
        self.connected = False
        
        # Video streaming
        self.video_running = False
        self.frame_callback: Optional[Callable] = None
        self.current_frame: Optional[np.ndarray] = None
        self.frame_lock = Lock()
        self.video_thread: Optional[Thread] = None
        
        # State
        self.current_state = TelloState()
        self.state_lock = Lock()
        self.state_thread: Optional[Thread] = None
        self.state_running = False
        
        print("[Tello Interface] Initialized")
    
    def connect(self) -> bool:
        """
        Connect to Tello drone
        
        Returns:
            True if successful
        """
        try:
            print("[Tello Interface] Connecting to Tello...")
            self.tello = Tello()
            self.tello.connect()
            
            # Get initial battery level
            battery = self.tello.get_battery()
            print(f"[Tello Interface] Connected! Battery: {battery}%")
            
            if battery < 10:
                print("[Tello Interface] WARNING: Battery critically low!")
            
            self.connected = True
            
            # Start state monitoring
            self._start_state_monitoring()
            
            return True
            
        except Exception as e:
            print(f"[Tello Interface] Connection failed: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """Disconnect from Tello"""
        print("[Tello Interface] Disconnecting...")
        
        # Stop video
        if self.video_running:
            self.stop_video_stream()
        
        # Stop state monitoring
        self._stop_state_monitoring()
        
        # End connection
        if self.tello:
            try:
                self.tello.end()
            except:
                pass
        
        self.connected = False
        print("[Tello Interface] Disconnected")
    
    # ========== Flight Commands ==========
    
    def takeoff(self) -> bool:
        """
        Takeoff
        
        Returns:
            True if successful
        """
        if not self.connected:
            print("[Tello Interface] Not connected!")
            return False
        
        try:
            print("[Tello Interface] Taking off...")
            self.tello.takeoff()
            with self.state_lock:
                self.current_state.is_flying = True
            print("[Tello Interface] Takeoff complete")
            return True
        except Exception as e:
            print(f"[Tello Interface] Takeoff failed: {e}")
            return False
    
    def land(self) -> bool:
        """
        Land
        
        Returns:
            True if successful
        """
        if not self.connected:
            print("[Tello Interface] Not connected!")
            return False
        
        try:
            print("[Tello Interface] Landing...")
            self.tello.land()
            with self.state_lock:
                self.current_state.is_flying = False
            print("[Tello Interface] Landing complete")
            return True
        except Exception as e:
            print(f"[Tello Interface] Landing failed: {e}")
            return False
    
    def emergency(self):
        """Emergency stop - cuts motors immediately"""
        if self.connected and self.tello:
            print("[Tello Interface] EMERGENCY STOP!")
            self.tello.emergency()
            with self.state_lock:
                self.current_state.is_flying = False
    
    def move(self, x: int, y: int, z: int, speed: int = 30):
        """
        Move relative to current position
        
        Args:
            x: Forward/backward in cm (20-500)
            y: Left/right in cm (20-500)
            z: Up/down in cm (20-500)
            speed: Speed in cm/s (10-100)
        """
        if not self.connected:
            print("[Tello Interface] Not connected!")
            return
        
        try:
            # Clamp values
            x = np.clip(x, -500, 500)
            y = np.clip(y, -500, 500)
            z = np.clip(z, -500, 500)
            speed = np.clip(speed, 10, 100)
            
            # Minimum movement is 20cm
            if abs(x) < 20 and abs(y) < 20 and abs(z) < 20:
                print("[Tello Interface] Movement too small (min 20cm)")
                return
            
            self.tello.go_xyz_speed(x, y, z, speed)
            print(f"[Tello Interface] Moving: x={x}, y={y}, z={z} at {speed}cm/s")
            
        except Exception as e:
            print(f"[Tello Interface] Move failed: {e}")
    
    def rotate(self, degrees: int):
        """
        Rotate by degrees
        
        Args:
            degrees: Degrees to rotate (positive = clockwise)
        """
        if not self.connected:
            print("[Tello Interface] Not connected!")
            return
        
        try:
            degrees = int(np.clip(degrees, -360, 360))
            
            if degrees > 0:
                self.tello.rotate_clockwise(degrees)
            else:
                self.tello.rotate_counter_clockwise(-degrees)
            
            print(f"[Tello Interface] Rotating {degrees} degrees")
            
        except Exception as e:
            print(f"[Tello Interface] Rotation failed: {e}")
    
    def send_rc_control(self, lr: int, fb: int, ud: int, yaw: int):
        """
        Send RC control commands
        
        Args:
            lr: Left/right (-100 to 100)
            fb: Forward/backward (-100 to 100)
            ud: Up/down (-100 to 100)
            yaw: Yaw rotation (-100 to 100)
        """
        if not self.connected:
            return
        
        try:
            # Clamp values
            lr = int(np.clip(lr, -100, 100))
            fb = int(np.clip(fb, -100, 100))
            ud = int(np.clip(ud, -100, 100))
            yaw = int(np.clip(yaw, -100, 100))
            
            self.tello.send_rc_control(lr, fb, ud, yaw)
            
        except Exception as e:
            print(f"[Tello Interface] RC control failed: {e}")
    
    # ========== Video Streaming ==========
    
    def start_video_stream(self, frame_callback: Optional[Callable] = None):
        """
        Start video streaming
        
        Args:
            frame_callback: Optional callback function(frame, timestamp)
        """
        if not self.connected:
            print("[Tello Interface] Not connected!")
            return
        
        if self.video_running:
            print("[Tello Interface] Video already running")
            return
        
        try:
            print("[Tello Interface] Starting video stream...")
            self.tello.streamon()
            self.frame_callback = frame_callback
            self.video_running = True
            
            # Start video thread
            self.video_thread = Thread(target=self._video_loop, daemon=True)
            self.video_thread.start()
            
            print("[Tello Interface] Video stream started")
            
        except Exception as e:
            print(f"[Tello Interface] Video start failed: {e}")
            self.video_running = False
    
    def stop_video_stream(self):
        """Stop video streaming"""
        if not self.video_running:
            return
        
        print("[Tello Interface] Stopping video stream...")
        self.video_running = False
        
        if self.video_thread:
            self.video_thread.join(timeout=2.0)
        
        if self.tello:
            try:
                self.tello.streamoff()
            except:
                pass
        
        print("[Tello Interface] Video stream stopped")
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Get current camera frame"""
        with self.frame_lock:
            return self.current_frame.copy() if self.current_frame is not None else None
    
    def _video_loop(self):
        """Video processing loop"""
        print("[Tello Interface] Video loop started")
        
        while self.video_running:
            try:
                frame = self.tello.get_frame_read().frame
                
                if frame is not None:
                    timestamp = time.time()
                    
                    # Store frame
                    with self.frame_lock:
                        self.current_frame = frame
                    
                    # Call callback
                    if self.frame_callback:
                        self.frame_callback(frame, timestamp)
                
                time.sleep(0.03)  # ~30 FPS
                
            except Exception as e:
                print(f"[Tello Interface] Video loop error: {e}")
                time.sleep(0.1)
        
        print("[Tello Interface] Video loop ended")
    
    # ========== State Monitoring ==========
    
    def _start_state_monitoring(self):
        """Start state monitoring thread"""
        self.state_running = True
        self.state_thread = Thread(target=self._state_loop, daemon=True)
        self.state_thread.start()
    
    def _stop_state_monitoring(self):
        """Stop state monitoring thread"""
        self.state_running = False
        if self.state_thread:
            self.state_thread.join(timeout=2.0)
    
    def _state_loop(self):
        """State monitoring loop"""
        print("[Tello Interface] State monitoring started")
        
        while self.state_running:
            try:
                if self.tello:
                    # Get all state values
                    state = TelloState(
                        # Position (Tello's mission pad coordinates if available)
                        x=self.tello.get_mission_pad_distance_x() or 0,
                        y=self.tello.get_mission_pad_distance_y() or 0,
                        z=self.tello.get_mission_pad_distance_z() or 0,
                        
                        # Orientation
                        pitch=self.tello.get_pitch(),
                        roll=self.tello.get_roll(),
                        yaw=self.tello.get_yaw(),
                        
                        # Velocities
                        vx=self.tello.get_speed_x(),
                        vy=self.tello.get_speed_y(),
                        vz=self.tello.get_speed_z(),
                        
                        # Accelerations
                        agx=self.tello.get_acceleration_x(),
                        agy=self.tello.get_acceleration_y(),
                        agz=self.tello.get_acceleration_z(),
                        
                        # Status
                        battery=self.tello.get_battery(),
                        temperature=self.tello.get_temperature(),
                        height=self.tello.get_height(),
                        time_of_flight=self.tello.get_distance_tof(),
                        barometer=self.tello.get_barometer(),
                        
                        # Timestamp
                        timestamp=time.time()
                    )
                    
                    with self.state_lock:
                        # Preserve flying state
                        state.is_flying = self.current_state.is_flying
                        self.current_state = state
                
                time.sleep(0.05)  # 20Hz state updates
                
            except Exception as e:
                print(f"[Tello Interface] State loop error: {e}")
                time.sleep(0.1)
        
        print("[Tello Interface] State monitoring ended")
    
    def get_state(self) -> Optional[TelloState]:
        """Get current Tello state"""
        with self.state_lock:
            return self.current_state
    
    # ========== Utility Functions ==========
    
    def get_battery(self) -> int:
        """Get battery percentage"""
        state = self.get_state()
        return state.battery if state else 0
    
    def get_height(self) -> int:
        """Get height in cm"""
        state = self.get_state()
        return state.height if state else 0


# Example usage
if __name__ == "__main__":
    def on_frame(frame, timestamp):
        """Frame callback example"""
        cv2.imshow("Tello Camera", frame)
        cv2.waitKey(1)
    
    # Initialize interface
    tello = TelloInterface()
    
    if tello.connect():
        try:
            # Start video
            tello.start_video_stream(frame_callback=on_frame)
            
            print("\nBattery:", tello.get_battery(), "%")
            print("Press Enter to takeoff, then Enter to land...")
            input()
            
            # Takeoff
            tello.takeoff()
            time.sleep(3)
            
            # Monitor state
            for i in range(20):
                state = tello.get_state()
                if state:
                    print(f"Height: {state.height}cm, Battery: {state.battery}%")
                time.sleep(0.5)
            
            input("Press Enter to land...")
            
            # Land
            tello.land()
            
        except KeyboardInterrupt:
            print("\nInterrupted!")
            tello.land()
        
        finally:
            tello.disconnect()
            cv2.destroyAllWindows()