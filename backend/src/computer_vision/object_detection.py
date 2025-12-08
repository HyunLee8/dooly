import cv2
from ultralytics import YOLO

def run_model(model_path="yolo11n.pt"):
    model = YOLO('yolo11n.pt')
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        return

    print(f"Using model: {model_path}")
    print("Press 'q' to quit.")

    try:
        while True:
            succcess, frame = cap.read()
            frame = cv2.flip(frame, 1)
            if not succcess:
                break
            results = model(frame, stream=True, device='mps')
            for r in results:
                frame = r.plot()
            cv2.imshow('YOLOv11n Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released and windows closed.")


if __name__ == "__main__":
    run_model()
