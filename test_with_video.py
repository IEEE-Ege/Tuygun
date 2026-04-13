"""
TEKNOFEST 2026 - Gerçek Video ile Pozisyon Tahmin Testi
========================================================
Kullanım:
    python3 test_with_video.py
 
Aynı klasörde şunlar olmalı:
    THYZ_2026_Ornek_Veri_1.MP4
    THYZ_2026_Ornek_Veri_1_translation.csv
    position_estimator.py
"""
 
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
 
from position_estimator import PositionEstimator, CAMERA_PROFILES
 
# ─── Ayarlar ──────────────────────────────────────────────────────────────────
 
VIDEO_PATH      = "THYZ_2026_Ornek_Veri_1.MP4"
CSV_PATH        = "THYZ_2026_Ornek_Veri_1_translation.csv"
GPS_FAIL_START  = 450       # ilk 450 kare GPS sağlıklı
GPS_FAIL_END    = None      # None = sonuna kadar bozuk
MAX_FRAMES      = 900       # hızlı test için 900, tam test için None
VIDEO_FPS       = 29.97     # videonun gerçek FPS'i
TARGET_FPS      = 7.5       # yarışmada kullanılacak FPS
FRAME_STEP      = int(round(VIDEO_FPS / TARGET_FPS))  # = 4 (her 4. kare)
 
# ─── Video + CSV Okuyucu ──────────────────────────────────────────────────────
 
def load_data(video_path, csv_path, max_frames, frame_step):
    """
    Videodan her FRAME_STEP karede bir kare alır.
    CSV'deki satır sırası ile eşleştirir.
    """
    cap = cv2.VideoCapture(video_path)
    df  = pd.read_csv(csv_path)
 
    if max_frames:
        df = df.head(max_frames)
 
    frames = []
    video_idx = 0  # videodaki gerçek kare numarası
 
    for csv_idx in range(len(df)):
        target = csv_idx * frame_step
        # Videoyu doğru kareye atla
        if target != video_idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            video_idx = target
 
        ret, frame = cap.read()
        if not ret:
            print(f"  Uyarı: kare {target} okunamadı, önceki kare kullanılıyor.")
            frames.append(frames[-1] if frames else np.zeros((1080,1920,3), dtype=np.uint8))
        else:
            frames.append(frame)
        video_idx += 1
 
        if csv_idx % 100 == 0:
            print(f"  Yükleniyor... {csv_idx}/{len(df)}")
 
    cap.release()
    print(f"  Toplam {len(frames)} kare yüklendi.")
    return frames, df
 
# ─── Test Koşucusu ────────────────────────────────────────────────────────────
 
def run_test():
    print("\n" + "="*55)
    print("  TEKNOFEST 2026 - Gerçek Video Testi")
    print("="*55)
 
    # Veri yükle
    print("\n[1/3] Video ve CSV yükleniyor...")
    frames, df = load_data(VIDEO_PATH, CSV_PATH, MAX_FRAMES, FRAME_STEP)
    n = len(frames)
    gps_fail_end = GPS_FAIL_END if GPS_FAIL_END else n
 
    print(f"\n[2/3] Optical flow çalıştırılıyor ({n} kare)...")
 
    cam       = CAMERA_PROFILES["rgb_1080p"]
    estimator = PositionEstimator(camera_profile=cam)
 
    ref_positions = []
    est_positions = []
    errors        = []
    gps_flags     = []
    flow_qualities = []  # her karedeki keypoint sayısı
 
    for i, (frame, row) in enumerate(zip(frames, df.itertuples())):
        ref = np.array([row.translation_x, row.translation_y, row.translation_z])
        gps_ok = not (GPS_FAIL_START <= i < gps_fail_end)
 
        tx, ty, tz = estimator.process(
            frame       = frame,
            gps_healthy = gps_ok,
            server_x    = float(ref[0]),
            server_y    = float(ref[1]),
            server_z    = float(ref[2]),
        )
 
        est = np.array([tx, ty, tz])
        err = float(np.linalg.norm(est - ref))
 
        ref_positions.append(ref.copy())
        est_positions.append(est.copy())
        errors.append(err)
        gps_flags.append(gps_ok)
 
        # Keypoint kalitesini takip et
        kp_count = len(estimator.prev_keypoints) if estimator.prev_keypoints is not None else 0
        flow_qualities.append(kp_count)
 
        if i % 100 == 0:
            tag = "GPS✓" if gps_ok else "OPT"
            print(f"  [{i:4d}/{n}] {tag}  "
                  f"ref=({ref[0]:7.2f},{ref[1]:7.2f})  "
                  f"est=({tx:7.2f},{ty:7.2f})  "
                  f"hata={err:.3f}m  kp={kp_count}")
 
    # ── İstatistikler ──────────────────────────────────────────────────────────
    errors       = np.array(errors)
    fail_mask    = ~np.array(gps_flags)
    ok_mask      =  np.array(gps_flags)
 
    print(f"\n[3/3] Sonuçlar:")
    print(f"{'─'*55}")
    print(f"  GPS sağlıklı   → ort hata : {errors[ok_mask].mean():.4f} m")
    if fail_mask.any():
        print(f"  GPS bozuk      → ort hata : {errors[fail_mask].mean():.4f} m")
        print(f"  GPS bozuk      → max hata : {errors[fail_mask].max():.4f} m")
        print(f"  GPS bozuk      → ilk 50k  : {errors[fail_mask][:50].mean():.4f} m")
    print(f"  Genel ort hata (Denklem 2): {errors.mean():.4f} m")
    print(f"  Ort keypoint sayısı        : {np.mean(flow_qualities):.0f}")
    print(f"{'='*55}\n")
 
    # ── Grafik ────────────────────────────────────────────────────────────────
    plot_results(ref_positions, est_positions, errors, gps_flags,
                 flow_qualities, GPS_FAIL_START, gps_fail_end)
 
    return errors
 
# ─── Grafik ───────────────────────────────────────────────────────────────────
 
def plot_results(refs, ests, errors, gps_flags, flow_qualities,
                 fail_start, fail_end):
    refs   = np.array(refs)
    ests   = np.array(ests)
    errs   = np.array(errors)
    flags  = np.array(gps_flags)
    xs     = np.arange(len(errs))
 
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle("TEKNOFEST 2026 – Gerçek Video Testi  [rgb_1080p]",
                 fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.42, wspace=0.32)
 
    fail_region = dict(xmin=fail_start, xmax=fail_end,
                       alpha=0.10, color="red")
 
    # 1. Kuş bakışı yörünge
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(refs[:,0], refs[:,1], "b-",  lw=1.2, label="Gerçek (GPS)", alpha=0.8)
    ax1.plot(ests[:,0], ests[:,1], "r--", lw=1.0, label="Tahmin (OF)",  alpha=0.8)
    ax1.set_title("Kuş bakışı yörünge (X-Y)")
    ax1.set_xlabel("X (m)"); ax1.set_ylabel("Y (m)")
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)
 
    # 2. Hata zaman serisi
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axvspan(**fail_region, label="GPS bozuk")
    ax2.plot(xs[flags],  errs[flags],  "g.", ms=2, label="GPS sağlıklı")
    ax2.plot(xs[~flags], errs[~flags], "r.", ms=2, label="Optical flow")
    ax2.set_title("Kare başına hata (Denklem 2)")
    ax2.set_xlabel("Kare"); ax2.set_ylabel("Hata (m)")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)
 
    # 3. X ekseni
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.axvspan(**fail_region)
    ax3.plot(xs, refs[:,0], "b-",  lw=1, label="Gerçek X", alpha=0.8)
    ax3.plot(xs, ests[:,0], "r--", lw=1, label="Tahmin X", alpha=0.8)
    ax3.set_title("X ekseni karşılaştırma")
    ax3.set_xlabel("Kare"); ax3.set_ylabel("X (m)")
    ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3)
 
    # 4. Y ekseni
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axvspan(**fail_region)
    ax4.plot(xs, refs[:,1], "b-",  lw=1, label="Gerçek Y", alpha=0.8)
    ax4.plot(xs, ests[:,1], "r--", lw=1, label="Tahmin Y", alpha=0.8)
    ax4.set_title("Y ekseni karşılaştırma")
    ax4.set_xlabel("Kare"); ax4.set_ylabel("Y (m)")
    ax4.legend(fontsize=8); ax4.grid(True, alpha=0.3)
 
    # 5. Z ekseni
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.axvspan(**fail_region)
    ax5.plot(xs, refs[:,2], "b-",  lw=1, label="Gerçek Z", alpha=0.8)
    ax5.plot(xs, ests[:,2], "r--", lw=1, label="Tahmin Z", alpha=0.8)
    ax5.set_title("Z ekseni karşılaştırma (irtifa)")
    ax5.set_xlabel("Kare"); ax5.set_ylabel("Z (m)")
    ax5.legend(fontsize=8); ax5.grid(True, alpha=0.3)
 
    # 6. Keypoint kalitesi
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axvspan(**fail_region)
    ax6.plot(xs, flow_qualities, "purple", lw=0.8, alpha=0.7)
    ax6.axhline(y=10, color="red", linestyle="--", lw=1, label="Min eşik (10)")
    ax6.set_title("Optical flow keypoint sayısı")
    ax6.set_xlabel("Kare"); ax6.set_ylabel("Keypoint sayısı")
    ax6.legend(fontsize=8); ax6.grid(True, alpha=0.3)
 
    out = "test_gercek_video_sonuc.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Grafik kaydedildi → {out}")
 
 
if __name__ == "__main__":
    run_test()