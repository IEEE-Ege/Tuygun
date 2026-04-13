"""
Scale ve yön değerlerini debug etmek için.
Kullanım: python3 debug_scale.py
"""
import cv2, numpy as np, pandas as pd
import sys
sys.path.insert(0, '.')
from position_estimator import PositionEstimator, CAMERA_PROFILES

VIDEO = "THYZ_2026_Ornek_Veri_1.MP4"
CSV   = "THYZ_2026_Ornek_Veri_1_translation.csv"
MAX   = 450   # Sadece GPS sağlıklı kısım
STEP  = 4

df  = pd.read_csv(CSV).head(MAX)
cap = cv2.VideoCapture(VIDEO)
cam = CAMERA_PROFILES["rgb_1080p"]
est = PositionEstimator(cam, assumed_altitude=50.0)

scale_log = []
angle_log = []

for i, row in df.iterrows():
    cap.set(cv2.CAP_PROP_POS_FRAMES, i * STEP)
    ret, frame = cap.read()
    if not ret: break

    tx, ty, tz = est.process(frame, True,
        float(row.translation_x), float(row.translation_y), float(row.translation_z))

    if i > 0 and i % 10 == 0:
        scale_log.append(est.scale)
        angle_log.append(est.world_angle_deg)

    if i % 50 == 0:
        print(f"[{i:4d}] scale={est.scale:.6f}  angle={est.world_angle_deg:.2f}deg  samples={est.n_scale_samples}")

cap.release()

print()
print("="*50)
print(f"Son scale  : {est.scale:.6f}")
print(f"Son angle  : {est.world_angle_deg:.2f} derece")
print(f"Scale samples: {est.n_scale_samples}")
if scale_log:
    print(f"Scale min  : {min(scale_log):.6f}")
    print(f"Scale max  : {max(scale_log):.6f}")
    print(f"Scale medyan: {float(np.median(scale_log)):.6f}")
print("="*50)
print()
print("Bu değerleri Claude'a yapıştır.")