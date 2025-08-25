
# train_model.py
"""
Trains a RandomForest on MediaPipe hand landmarks extracted from your images.

Data layout expected:
  data/
    HELLO/
      img1.jpg, img2.jpg, ...
    YES/
      ...

Run:
  python train_model.py
"""
import os, cv2, numpy as np, joblib, json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import mediapipe as mp

DATA_DIR = "data"
MODEL_PATH = "sign_model.pkl"
LABELS_PATH = "labels.json"

mp_hands = mp.solutions.hands

def extract_features_from_image(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
        res = hands.process(img_rgb)
    if not res.multi_hand_landmarks:
        return None
    lm = res.multi_hand_landmarks[0]
    coords = np.array([[p.x, p.y, p.z] for p in lm.landmark], dtype=np.float32)  # (21,3)
    wrist = coords[0].copy()
    coords -= wrist
    max_xy = np.max(np.abs(coords[:, :2])) or 1.0
    coords[:, :2] /= max_xy
    max_z = np.max(np.abs(coords[:, 2])) or 1.0
    coords[:, 2] /= max_z
    return coords.flatten()  # (63,)

def load_dataset():
    X, y = [], []
    for label in sorted(os.listdir(DATA_DIR)):
        folder = os.path.join(DATA_DIR, label)
        if not os.path.isdir(folder):
            continue
        for fname in os.listdir(folder):
            if not fname.lower().endswith((".jpg",".jpeg",".png",".bmp")):
                continue
            path = os.path.join(folder, fname)
            img = cv2.imread(path)
            if img is None:
                continue
            feat = extract_features_from_image(img)
            if feat is not None:
                X.append(feat)
                y.append(label)
    return np.array(X), np.array(y)

def main():
    X, y = load_dataset()
    if len(X) == 0:
        print("❌ No samples found. Add images into data/<label>/ and ensure a hand is visible.")
        return

    if len(X) >= 10 and len(np.unique(y)) > 1:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    else:
        X_train, y_train = X, y
        X_test, y_test = X[:0], y[:0]

    clf = RandomForestClassifier(n_estimators=250, random_state=42)
    clf.fit(X_train, y_train)

    if len(X_test) > 0:
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"✅ Accuracy: {acc:.3f}")
        print(classification_report(y_test, y_pred))
    else:
        print("⚠️ Skipping test split (dataset too small).")

    joblib.dump(clf, MODEL_PATH)
    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        import json
        json.dump(sorted(list(set(y_train.tolist()))), f, indent=2)
    print(f"💾 Saved model to {MODEL_PATH} and labels to {LABELS_PATH}")

if __name__ == "__main__":
    main()
