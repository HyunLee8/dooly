"""
Basic DJI Tello Drone Controller
Handles connection and basic flight commands
"""
from djitellopy import Tello
import time

class TelloController:
    def __init__(self):
        """Initialize connection to Tello drone"""
        self.drone = Tello()
        self.is_flying = False
        
    def connect(self):
        """Connect to the drone"""
        try:
            print("[Connecting...] Attempting to connect to Tello...")
            self.drone.connect()
            battery = self.drone.get_battery()
            print(f"[Connected!] Battery: {battery}%")
            
            if battery < 10:
                print("[Warning!] Battery too low for flight!")
                return False
            return True
        except Exception as e:
            print(f"[Connection Error] {e}")
            return False
    
    def takeoff(self):
        """Take off"""
        if not self.is_flying:
            print("[Taking Off...]")
            self.drone.takeoff()
            self.is_flying = True
            time.sleep(2)  # Wait for stable hover
            print("[Airborne!]")
    
    def land(self):
        """Land the drone"""
        if self.is_flying:
            print("[Landing...]")
            self.drone.land()
            self.is_flying = False
            print("[Landed!]")
    
    def move_forward(self, distance=30):
        """Move forward (distance in cm, 20-500)"""
        print(f"[Moving Forward] {distance}cm")
        self.drone.move_forward(distance)
    
    def move_back(self, distance=30):
        """Move backward (distance in cm, 20-500)"""
        print(f"[Moving Back] {distance}cm")
        self.drone.move_back(distance)
    
    def move_left(self, distance=30):
        """Move left (distance in cm, 20-500)"""
        print(f"[Moving Left] {distance}cm")
        self.drone.move_left(distance)
    
    def move_right(self, distance=30):
        """Move right (distance in cm, 20-500)"""
        print(f"[Moving Right] {distance}cm")
        self.drone.move_right(distance)
    
    def move_up(self, distance=30):
        """Move up (distance in cm, 20-500)"""
        print(f"[Moving Up] {distance}cm")
        self.drone.move_up(distance)
    
    def move_down(self, distance=30):
        """Move down (distance in cm, 20-500)"""
        print(f"[Moving Down] {distance}cm")
        self.drone.move_down(distance)
    
    def rotate_clockwise(self, degrees=90):
        """Rotate clockwise (degrees, 1-360)"""
        print(f"[Rotating Clockwise] {degrees}°")
        self.drone.rotate_clockwise(degrees)
    
    def rotate_counter_clockwise(self, degrees=90):
        """Rotate counter-clockwise (degrees, 1-360)"""
        print(f"[Rotating Counter-Clockwise] {degrees}°")
        self.drone.rotate_counter_clockwise(degrees)
    
    def flip_forward(self):
        """Perform forward flip"""
        print("[Flipping Forward!]")
        self.drone.flip_forward()
    
    def flip_back(self):
        """Perform backward flip"""
        print("[Flipping Backward!]")
        self.drone.flip_back()
    
    def get_battery(self):
        """Get current battery percentage"""
        return self.drone.get_battery()
    
    def emergency_stop(self):
        """Emergency stop - kills motors immediately"""
        print("[EMERGENCY STOP!]")
        self.drone.emergency()
        self.is_flying = False
    
    def disconnect(self):
        """Safely disconnect from drone"""
        if self.is_flying:
            self.land()
        print("[Disconnecting...]")
        self.drone.end()


# Example usage
if __name__ == "__main__":
    controller = TelloController()
    
    if controller.connect():
        try:
            # Simple flight test
            controller.takeoff()
            time.sleep(2)
            
            controller.move_forward(50)
            time.sleep(2)
            
            controller.rotate_clockwise(360)
            time.sleep(2)
            
            controller.land()
            
        except KeyboardInterrupt:
            print("\n[Interrupted!] Landing drone...")
            controller.land()
        except Exception as e:
            print(f"[Error] {e}")
            controller.emergency_stop()
        finally:
            controller.disconnect()
    else:
        print("[Failed] Could not connect to drone")