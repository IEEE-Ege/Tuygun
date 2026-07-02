import cv2
import os
import pandas as pd
import numpy as np
from mono_vo import MonocularVO

# ── Veri Kaynağı ────────────────────────────────────────────────────────────
DATA_DIR      = "2025/THYZ_2025_Oturum_2-2"
CSV_PATH      = "2025/THYZ_2025_Oturum_2_Translation.csv"
DEBUG_FRAMES  = 50        # kaç kare incelensin
PROCESS_SCALE = 0.5       # görüntü küçültme (0.5 = 1920×1080)
# ─────────────────────────────────────────────────────────────────────────────

SCALE_W = (3840 / 4000) * PROCESS_SCALE
SCALE_H = (2160 / 3000) * PROCESS_SCALE
FX = 2792.2 * SCALE_W
FY = 2795.2 * SCALE_H
CX = 1988.0 * SCALE_W
CY = 1562.2 * SCALE_H

# ── Kalibrasyon & VO ─────────────────────────────────────────────────────────
K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]], dtype=np.float64)
vo = MonocularVO()
vo.K = K
vo.update_calibration(focal=FX, pp=(CX, CY))

# ── Veri Seti ────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
df["frame_path"] = df["frame_numbers"].apply(
    lambda fn: os.path.join(DATA_DIR, fn + ".webp")
)

init_x = float(df.iloc[0]["translation_x"])
init_y = float(df.iloc[0]["translation_y"])

frame_idx = 0

for _, row in df.iterrows():
    if frame_idx >= DEBUG_FRAMES:
        break

    # ── Kareyi Oku ───────────────────────────────────────────────────────────
    frame = cv2.imread(row["frame_path"])
    if frame is None:
        print(f"UYARI: {row['frame_path']} okunamadı")
        frame_idx += 1
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=PROCESS_SCALE, fy=PROCESS_SCALE,
                      interpolation=cv2.INTER_AREA)

    true_x = float(row["translation_x"])
    true_y = float(row["translation_y"])
    true_z = float(row["translation_z"])

    json_data = {
        "translation_x": true_x,
        "translation_y": true_y,
        "translation_z": true_z,
        "gps_health_status": 1,   # debug: GPS hep sağlıklı
    }

    # ── İlk Kare: process_frame ile başlat (initial_gps doğru kurulsun) ─────
    if not vo.is_initialized:
        sonuc = vo.process_frame(gray, json_data)
        print(f"Kare {frame_idx:3d}: INIT | Feature sayısı: {len(vo.px_ref)}")
        frame_idx += 1
        continue

    # ── Sonraki Kareler: iç logu göster, ardından process_frame çağır ────────
    px_ref_tmp, px_cur_tmp = vo.feature_tracking(vo.prev_frame, gray, vo.px_ref)

    if len(px_ref_tmp) >= 8:
        px_cur_c = px_cur_tmp - np.array(vo.pp)
        px_ref_c = px_ref_tmp - np.array(vo.pp)
        m, inliers = cv2.estimateAffinePartial2D(
            px_cur_c, px_ref_c,
            method=cv2.RANSAC, ransacReprojThreshold=3.0, maxIters=2000, confidence=0.99
        )
        if m is not None:
            dx = m[0, 2]; dy = m[1, 2]
            yaw = -np.degrees(np.arctan2(m[1, 0], m[0, 0]))
            pixel_disp = np.sqrt(dx**2 + dy**2)
            gps_scale = vo.get_absolute_scale(json_data, vo.keyframe_json, pixel_disp)
            inlier_cnt = int(np.sum(inliers)) if inliers is not None else 0
            print(
                f"Kare {frame_idx:3d}: "
                f"dx={dx:8.2f}px  dy={dy:8.2f}px  |  "
                f"yaw={yaw:7.3f}°  |  "
                f"pix_disp={pixel_disp:6.2f}px  |  "
                f"GPS_scale={gps_scale:.5f}m  |  "
                f"PPM={vo.pixel_to_meter:.5f}  |  "
                f"inliers={inlier_cnt}/{len(px_cur_tmp)}"
            )
        else:
            print(f"Kare {frame_idx:3d}: Affine BAŞARISIZ")
    else:
        print(f"Kare {frame_idx:3d}: YETERSİZ FEATURE ({len(px_ref_tmp)})")

    # Gerçek hesap
    sonuc = vo.process_frame(gray, json_data)
    px = sonuc["detected_translations"][0]["translation_x"]
    py = sonuc["detected_translations"][0]["translation_y"]
    gt_rx = true_x - init_x
    gt_ry = true_y - init_y
    err = np.sqrt((gt_rx - px)**2 + (gt_ry - py)**2)
    print(f"           GT:({gt_rx:.4f},{gt_ry:.4f})  VO:({px:.4f},{py:.4f})  Hata:{err:.4f}m")
    print()

    frame_idx += 1
