
# verify_dataset.py
"""
Prints how many images are present per label in data/.
Run: python verify_dataset.py
"""
import os
from collections import defaultdict

DATA_DIR = "data"
counts = defaultdict(int)

for label in sorted(os.listdir(DATA_DIR)):
    folder = os.path.join(DATA_DIR, label)
    if not os.path.isdir(folder):
        continue
    for f in os.listdir(folder):
        if f.lower().endswith((".jpg",".jpeg",".png",".bmp")):
            counts[label] += 1

if not counts:
    print("No images found. Use capture_images.py or add your own images into data/<label>/")
else:
    print("Images per label:")
    for k, v in counts.items():
        print(f"  {k}: {v}")
