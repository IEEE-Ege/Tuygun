"""
TEKNOFEST 2026 - Pozisyon Tahmini Test Scripti
===============================================
Gerçek görüntü olmadan da çalışır:
  - CSV'deki referans pozisyonları kullanır
  - Her kare için sentetik (yapay) bir hava görüntüsü üretir
  - Optical flow ile pozisyon tahmin eder
  - Şartname Denklem 2 ile hata hesaplar
  - Sonuç grafiği PNG olarak kaydeder

Kullanım:
  pip install pandas numpy opencv-python matplotlib
  python test_position.py
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# position_estimator.py ile aynı klasörde olmalı
from position_estimator import PositionEstimator, CAMERA_PROFILES


# ─── Sentetik Görüntü Üretici ─────────────────────────────────────────────────

class SyntheticSceneGenerator:
    """
    Gerçekçi bir hava görüntüsü simüle eder.
    Zemin dokusu sabit, kamera her karede tx/ty kadar kayar.
    Bu sayede optical flow gerçek bir hareketi takip eder.
    """

    def __init__(self, width=1920, height=1080, altitude=50.0):
        self.W = width
        self.H = height
        self.altitude = altitude

        # Büyük bir zemin dokusu oluştur (kamera kayarken kenar görmemek için)
        margin = 800
        self.ground_w = width  + margin * 2
        self.ground_h = height + margin * 2
        self.offset_x = margin
        self.offset_y = margin

        # Perlin-benzeri gürültüden zemin dokusu
        rng = np.random.default_rng(42)
        base = rng.integers(30, 220, (self.ground_h // 4, self.ground_w // 4), dtype=np.uint8)
        self.ground = cv2.resize(base, (self.ground_w, self.ground_h), interpolation=cv2.INTER_CUBIC)

        # Bazı "yapılar" ekle (dikdörtgenler)
        for _ in range(120):
            x = rng.integers(0, self.ground_w - 80)
            y = rng.integers(0, self.ground_h - 80)
            w = rng.integers(20, 80)
            h = rng.integers(20, 80)
            c = int(rng.integers(40, 200))
            cv2.rectangle(self.ground, (x, y), (x+w, y+h), c, -1)

        # Kamera pozisyonu (piksel cinsinden zemin üzerinde)
        self.cam_x = float(self.offset_x)
        self.cam_y = float(self.offset_y)

    def move_camera(self, dx_meter: float, dy_meter: float, fx: float, altitude: float):
        """
        Gerçek dünya hareketini (metre) piksel hareketine çevir ve kamerayı taşı.
        """
        dx_px = (dx_meter / altitude) * fx
        dy_px = (dy_meter / altitude) * fx
        self.cam_x += dx_px
        self.cam_y += dy_px

    def get_frame(self) -> np.ndarray:
        """Mevcut kamera pozisyonundan görüntü al."""
        x1 = max(0, int(self.cam_x))
        y1 = max(0, int(self.cam_y))
        x2 = x1 + self.W
        y2 = y1 + self.H

        # Sınır kontrolü
        x2 = min(x2, self.ground_w)
        y2 = min(y2, self.ground_h)
        x1 = x2 - self.W
        y1 = y2 - self.H

        crop = self.ground[y1:y2, x1:x2]
        frame_bgr = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)

        # Hafif gürültü ekle (gerçekçilik için)
        noise = np.random.normal(0, 4, frame_bgr.shape).astype(np.int16)
        frame_bgr = np.clip(frame_bgr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return frame_bgr


# ─── Test Koşucusu ────────────────────────────────────────────────────────────

def run_test(csv_path: str,
             gps_fail_start: int = 450,
             gps_fail_end:   int = None,
             max_frames:     int = 900,
             camera_name:    str = "rgb_1080p"):
    """
    csv_path       : THYZ_2026_Ornek_Veri_1_translation.csv
    gps_fail_start : GPS'in bozulmaya başladığı kare indeksi
    gps_fail_end   : GPS'in düzeldiği kare (None = sonuna kadar)
    max_frames     : test edilecek maksimum kare (hızlı test için düşür)
    camera_name    : "thermal" | "rgb_4k" | "rgb_1080p"
    """
    df = pd.read_csv(csv_path).head(max_frames)
    n  = len(df)
    if gps_fail_end is None or gps_fail_end > n:
        gps_fail_end = n

    cam      = CAMERA_PROFILES[camera_name]
    scene    = SyntheticSceneGenerator(
        width    = cam["width"],
        height   = cam["height"],
        altitude = 50.0
    )
    estimator = PositionEstimator(camera_profile=cam)

    ref_positions  = []   # gerçek GPS
    est_positions  = []   # bizim tahminimiz
    errors         = []
    gps_flags      = []

    print(f"\n{'='*55}")
    print(f"  Kamera        : {camera_name}")
    print(f"  Toplam kare   : {n}")
    print(f"  GPS bozuk     : [{gps_fail_start} – {gps_fail_end}]")
    print(f"{'='*55}")

    prev_ref = np.array([0.0, 0.0, 0.0])

    for i, row in df.iterrows():
        ref = np.array([row["translation_x"],
                        row["translation_y"],
                        row["translation_z"]])

        # Gerçek hareketi sahneye uygula (sentetik görüntü oluşsun)
        delta = ref - prev_ref
        scene.move_camera(delta[0], delta[1], cam["fx"], altitude=50.0)
        frame = scene.get_frame()
        prev_ref = ref.copy()

        gps_ok  = not (gps_fail_start <= i < gps_fail_end)
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

        if i % 100 == 0:
            tag = "GPS✓" if gps_ok else "OPT"
            print(f"  [{i:4d}/{n}] {tag}  ref=({ref[0]:7.2f},{ref[1]:7.2f})  "
                  f"est=({tx:7.2f},{ty:7.2f})  hata={err:.3f}m")

    # ── İstatistikler ──────────────────────────────────────────────────────────
    errors      = np.array(errors)
    fail_mask   = ~np.array(gps_flags)
    ok_mask     =  np.array(gps_flags)

    print(f"\n{'─'*55}")
    print(f"  GPS sağlıklı   → ort hata : {errors[ok_mask].mean():.4f} m  (sıfır olmalı)")
    if fail_mask.any():
        print(f"  GPS bozuk      → ort hata : {errors[fail_mask].mean():.4f} m")
        print(f"  GPS bozuk      → max hata : {errors[fail_mask].max():.4f} m")
    print(f"  Genel ort hata (Denklem 2): {errors.mean():.4f} m")
    print(f"{'='*55}\n")

    # ── Grafik ────────────────────────────────────────────────────────────────
    _plot_results(ref_positions, est_positions, errors, gps_flags, camera_name)

    return errors


def _plot_results(refs, ests, errors, gps_flags, camera_name):
    refs  = np.array(refs)
    ests  = np.array(ests)
    errs  = np.array(errors)
    flags = np.array(gps_flags)
    xs    = np.arange(len(errs))

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f"TEKNOFEST 2026 – Pozisyon Tahmin Testi  [{camera_name}]",
                 fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

    # 1. Kuş bakışı yörünge
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(refs[:, 0], refs[:, 1], "b-",  lw=1.2, label="Gerçek (GPS)", alpha=0.8)
    ax1.plot(ests[:, 0], ests[:, 1], "r--", lw=1.0, label="Tahmin (OF)",  alpha=0.8)
    ax1.set_title("Kuş Bakışı Yörünge (X-Y)")
    ax1.set_xlabel("X (m)"); ax1.set_ylabel("Y (m)")
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

    # 2. Hata zaman serisi
    ax2 = fig.add_subplot(gs[0, 1])
    fail_xs = xs[~flags]
    ok_xs   = xs[flags]
    ax2.fill_betweenx([0, errs.max()*1.1],
                      fail_xs[0] if len(fail_xs) else 0,
                      fail_xs[-1] if len(fail_xs) else 0,
                      alpha=0.12, color="red", label="GPS bozuk")
    ax2.plot(xs[flags],  errs[flags],  "g.", ms=2, label="GPS sağlıklı")
    ax2.plot(xs[~flags], errs[~flags], "r.", ms=2, label="Optical Flow")
    ax2.set_title("Kare Başına Hata (Denklem 2)")
    ax2.set_xlabel("Kare"); ax2.set_ylabel("Hata (m)")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

    # 3. X ekseni karşılaştırma
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(xs, refs[:, 0], "b-",  lw=1, label="Gerçek X", alpha=0.8)
    ax3.plot(xs, ests[:, 0], "r--", lw=1, label="Tahmin X", alpha=0.8)
    ax3.axvspan(fail_xs[0] if len(fail_xs) else 0,
                fail_xs[-1] if len(fail_xs) else 0,
                alpha=0.08, color="red")
    ax3.set_title("X Ekseni Karşılaştırma")
    ax3.set_xlabel("Kare"); ax3.set_ylabel("X (m)")
    ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3)

    # 4. Y ekseni karşılaştırma
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(xs, refs[:, 1], "b-",  lw=1, label="Gerçek Y", alpha=0.8)
    ax4.plot(xs, ests[:, 1], "r--", lw=1, label="Tahmin Y", alpha=0.8)
    ax4.axvspan(fail_xs[0] if len(fail_xs) else 0,
                fail_xs[-1] if len(fail_xs) else 0,
                alpha=0.08, color="red")
    ax4.set_title("Y Ekseni Karşılaştırma")
    ax4.set_xlabel("Kare"); ax4.set_ylabel("Y (m)")
    ax4.legend(fontsize=8); ax4.grid(True, alpha=0.3)

    out = "test_sonuc.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Grafik kaydedildi → {out}")


# ─── Giriş Noktası ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    CSV = "THYZ_2026_Ornek_Veri_1_translation.csv"

    run_test(
        csv_path       = CSV,
        gps_fail_start = 450,    # ilk 450 kare GPS sağlıklı
        gps_fail_end   = None,   # sonuna kadar bozuk
        max_frames     = 900,    # hızlı test: 900 kare (~2 dk)
                                 # tam test için: max_frames=9022
        camera_name    = "rgb_1080p",
    )
