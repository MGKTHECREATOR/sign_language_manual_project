
# test_camera.py
"""
Quick camera test: opens webcam and shows live video.
Run: python test_camera.py
"""
import cv2, argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera_index", type=int, default=0)
    args = parser.parse_args()
    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        print("❌ Could not open camera.")
        return
    print("✅ Camera opened. Press Q to quit.")
    while True:
        ok, frame = cap.read()
        if not ok: break
        frame = cv2.flip(frame, 1)
        cv2.imshow("Camera Test", frame)
        if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
