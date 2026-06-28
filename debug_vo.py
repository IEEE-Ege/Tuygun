import cv2
import pandas as pd
import numpy as np
from mono_vo import MonocularVO

# Kalibrasyon
fx, fy, cx, cy = 1389.7, 1387.1, 954.007, 558.896
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

vo = MonocularVO()
vo.K = K
vo.update_calibration(focal=fx, pp=(cx, cy))

df = pd.read_csv('THYZ_2026_Ornek_Veri_1_translation.csv')
cap = cv2.VideoCapture('THYZ_2026_Ornek_Veri_1.MP4')

frame_idx = 0
while cap.isOpened() and frame_idx < 50:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    row = df.iloc[frame_idx]
    true_x = float(row['translation_x'])
    true_y = float(row['translation_y'])
    true_z = float(row['translation_z'])
    
    json_data = {
        "translation_x": true_x,
        "translation_y": true_y,
        "translation_z": true_z,
        "gps_health_status": 1
    }
    
    # VO icin process_frame cagirmadan once bazi internal debug bilgileri alalim
    vo.cur_frame = gray
    
    if not vo.is_initialized:
        vo.px_ref = vo.feature_detection(vo.cur_frame)
        vo.prev_frame = vo.cur_frame.copy()
        vo.prev_json_data = json_data
        vo.is_initialized = True
        print(f"Kare {frame_idx:3d}: INIT | Feature sayisi: {len(vo.px_ref)}")
        frame_idx += 1
        continue
    
    # Feature tracking
    px_ref_before = len(vo.px_ref)
    vo.px_ref, vo.px_cur = vo.feature_tracking(vo.prev_frame, vo.cur_frame, vo.px_ref)
    
    if len(vo.px_ref) < 8:
        print(f"Kare {frame_idx:3d}: YETERSIZ FEATURE ({len(vo.px_ref)})")
        vo.px_cur = vo.feature_detection(vo.cur_frame)
        vo.px_ref = vo.px_cur
        vo.prev_frame = vo.cur_frame.copy()
        vo.prev_json_data = json_data
        frame_idx += 1
        continue
    
    # Affine hesapla
    px_cur_c = vo.px_cur - np.array(vo.pp)
    px_ref_c = vo.px_ref - np.array(vo.pp)
    
    m, inliers = cv2.estimateAffinePartial2D(
        px_cur_c, px_ref_c,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=2000,
        confidence=0.99
    )
    
    if m is not None:
        dx = m[0, 2]
        dy = m[1, 2]
        yaw = -np.arctan2(m[1, 0], m[0, 0])
        scale_from_affine = np.sqrt(m[0,0]**2 + m[1,0]**2)  # zoom/scale factor
        
        # GPS scale
        scale = vo.get_absolute_scale(json_data, vo.prev_json_data)
        
        # Debug sonucu
        sonuc = vo.process_frame(gray, json_data)  # gercek hesabi da yap
        px = sonuc["detected_translations"][0]["translation_x"]
        py = sonuc["detected_translations"][0]["translation_y"]
        
        inlier_count = np.sum(inliers) if inliers is not None else 0
        
        print(f"Kare {frame_idx:3d}: dx={dx:8.2f}px, dy={dy:8.2f}px | yaw={np.degrees(yaw):7.3f}deg | "
              f"affine_scale={scale_from_affine:.4f} | GPS_scale={scale:.6f} | "
              f"inliers={inlier_count}/{len(vo.px_cur)} | "
              f"GT=({true_x:.4f},{true_y:.4f}) -> VO=({px:.2f},{py:.2f})")
    else:
        sonuc = vo.process_frame(gray, json_data)
        print(f"Kare {frame_idx:3d}: Affine BASARISIZ")
    
    frame_idx += 1

cap.release()
