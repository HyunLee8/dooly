"""
Main entry point for DJI Tello drone control
"""
from tello_controller import TelloController
import time

def simple_test_flight(controller):
    """Perform a simple test flight"""
    print("\n=== Starting Simple Test Flight ===\n")
    
    controller.takeoff()
    time.sleep(2)
    
    # Square pattern
    print("Flying in a square...")
    controller.move_forward(50)
    time.sleep(1)
    controller.rotate_clockwise(90)
    time.sleep(1)
    
    controller.move_forward(50)
    time.sleep(1)
    controller.rotate_clockwise(90)
    time.sleep(1)
    
    controller.move_forward(50)
    time.sleep(1)
    controller.rotate_clockwise(90)
    time.sleep(1)
    
    controller.move_forward(50)
    time.sleep(1)
    controller.rotate_clockwise(90)
    time.sleep(1)
    
    controller.land()
    print("\n=== Test Flight Complete ===\n")

def manual_control(controller):
    """Manual keyboard control"""
    print("\n=== Manual Control Mode ===")
    print("Commands:")
    print("  t - takeoff")
    print("  l - land")
    print("  w - forward")
    print("  s - backward")
    print("  a - left")
    print("  d - right")
    print("  u - up")
    print("  j - down")
    print("  q - rotate left")
    print("  e - rotate right")
    print("  b - check battery")
    print("  x - emergency stop")
    print("  exit - quit")
    print()
    
    while True:
        cmd = input("Enter command: ").strip().lower()
        
        if cmd == 'exit':
            if controller.is_flying:
                controller.land()
            break
        elif cmd == 't':
            controller.takeoff()
        elif cmd == 'l':
            controller.land()
        elif cmd == 'w':
            controller.move_forward(30)
        elif cmd == 's':
            controller.move_back(30)
        elif cmd == 'a':
            controller.move_left(30)
        elif cmd == 'd':
            controller.move_right(30)
        elif cmd == 'u':
            controller.move_up(30)
        elif cmd == 'j':
            controller.move_down(30)
        elif cmd == 'q':
            controller.rotate_counter_clockwise(45)
        elif cmd == 'e':
            controller.rotate_clockwise(45)
        elif cmd == 'b':
            battery = controller.get_battery()
            print(f"Battery: {battery}%")
        elif cmd == 'x':
            controller.emergency_stop()
        else:
            print("Unknown command")

def main():
    """Main function"""
    controller = TelloController()
    
    if not controller.connect():
        print("Failed to connect. Make sure:")
        print("  1. Tello is powered on")
        print("  2. You're connected to Tello's WiFi (TELLO-XXXXXX)")
        print("  3. No other programs are using the drone")
        return
    
    try:
        print("\nSelect mode:")
        print("  1 - Simple test flight")
        print("  2 - Manual keyboard control")
        choice = input("Enter choice (1 or 2): ").strip()
        
        if choice == '1':
            simple_test_flight(controller)
        elif choice == '2':
            manual_control(controller)
        else:
            print("Invalid choice")
            
    except KeyboardInterrupt:
        print("\n[Interrupted!] Landing drone...")
        if controller.is_flying:
            controller.land()
    except Exception as e:
        print(f"[Error] {e}")
        controller.emergency_stop()
    finally:
        controller.disconnect()
        print("Program ended")

if __name__ == "__main__":
    main()