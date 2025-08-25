
# detect_sign.py
"""
Opens webcam, detects a hand, predicts the sign using the trained model,
shows the predicted label on the screen with smoothing.

Run:
  python detect_sign.py
  python detect_sign.py --camera_index 1
"""
import cv2, numpy as np, joblib, argparse, collections
import mediapipe as mp

MODEL_PATH = "sign_model.pkl"

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

def extract_features_from_landmarks(lm):
    coords = np.array([[p.x, p.y, p.z] for p in lm.landmark], dtype=np.float32)  # (21,3)
    wrist = coords[0].copy()
    coords -= wrist
    max_xy = np.max(np.abs(coords[:, :2])) or 1.0
    coords[:, :2] /= max_xy
    max_z = np.max(np.abs(coords[:, 2])) or 1.0
    coords[:, 2] /= max_z
    return coords.flatten()  # (63,)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera_index", type=int, default=0)
    parser.add_argument("--smooth", type=int, default=15, help="frames to smooth predictions")
    args = parser.parse_args()

    try:
        clf = joblib.load(MODEL_PATH)
    except Exception as e:
        print("❌ Could not load model. Run: python train_model.py")
        print(e)
        return

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        print("❌ Could not open camera. Try a different --camera_index.")
        return

    buffer = collections.deque(maxlen=args.smooth)
    with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            text = "No hand"
            if res.multi_hand_landmarks:
                lm = res.multi_hand_landmarks[0]
                feats = extract_features_from_landmarks(lm).reshape(1, -1)
                pred = clf.predict(feats)[0]
                buffer.append(pred)
                mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
                # majority vote smoothing
                counts = {}
                for p in buffer:
                    counts[p] = counts.get(p, 0) + 1
                text = max(counts, key=counts.get)

            cv2.putText(frame, f"Detected: {text}", (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.putText(frame, "Q=quit", (10,75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.imshow("Sign Detection (Live)", frame)
            if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
