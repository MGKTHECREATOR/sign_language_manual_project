
# capture_images.py
"""
A simple tool to CAPTURE your own dataset manually.

Usage (Terminal):
  python capture_images.py --label HELLO --count 300
  python capture_images.py --label YES --count 300 --camera_index 1

Keys:
  q  -> quit
  space -> save the current frame (one image)

It saves images into: data/<LABEL>/
"""
import cv2, os, argparse, time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True, help="Label name, e.g., HELLO")
    parser.add_argument("--count", type=int, default=300, help="Target number of images to save")
    parser.add_argument("--out_dir", default="data", help="Dataset root")
    parser.add_argument("--camera_index", type=int, default=0, help="cv2.VideoCapture index")
    args = parser.parse_args()

    save_dir = os.path.join(args.out_dir, args.label)
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        print("❌ Could not open camera. Try a different --camera_index.")
        return

    saved = len([f for f in os.listdir(save_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))])
    print(f"📸 Starting capture for '{args.label}'. Already have {saved}. Target {args.count}.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("❌ Failed to read from camera.")
            break

        frame = cv2.flip(frame, 1)  # mirror
        cv2.putText(frame, f"Label: {args.label}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.putText(frame, f"Saved: {saved}/{args.count}", (10,65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.putText(frame, "SPACE=save   Q=quit", (10,100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.imshow("Capture Images", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == 32:  # space
            ts = int(time.time()*1000)
            path = os.path.join(save_dir, f"{args.label}_{ts}.jpg")
            cv2.imwrite(path, frame)
            saved += 1
            if saved >= args.count:
                print("✅ Reached target count.")
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"✅ Finished. Images saved to: {save_dir}")

if __name__ == "__main__":
    main()
