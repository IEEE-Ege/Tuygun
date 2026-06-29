import cv2
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mono_vo import MonocularVO
import time
import math

# ── Yapılandırma — İSTENEN VİDEOYU BURADAN SEÇ ───────────────────────────────
#
# SEÇENEK A: 2025 verisi (GPS referansı mevcut)
# DATA_DIR            = "2025/THYZ_2025_Oturum_2-2"
# CSV_PATH            = "2025/THYZ_2025_Oturum_2_Translation.csv"
# OUT_FIG             = "trajectory_comparison_2025.png"
# GPS_KESILME_KARESI  = 300
# PROCESS_SCALE       = 0.5
# FX = 2792.2 * (3840/4000) * PROCESS_SCALE
# FY = 2795.2 * (2160/3000) * PROCESS_SCALE
# CX = 1988.0 * (3840/4000) * PROCESS_SCALE
# CY = 1562.2 * (2160/3000) * PROCESS_SCALE
#
# SEÇENEK B: 2024 Video 3 (GPS referansı yok — salt optik akış testi)
# DATA_DIR            = "2024/TUYZ_Video_3_v5y9z"
# CSV_PATH            = "2024/TUYZ_Video_3_v5y9z.csv"
# OUT_FIG             = "trajectory_comparison_2024_v3.png"
# GPS_KESILME_KARESI  = 0
#
# SEÇENEK C: 2024 Video 5 (GPS verisi mevcut)
# DATA_DIR            = "TUYZ_Video_5_igwak"
# CSV_PATH            = "TUYZ_Video_5_igwak.csv"
# OUT_FIG             = "trajectory_comparison_2024_v5.png"
# GPS_KESILME_KARESI  = 300
#
# SEÇENEK E: 2026 Video 1 (GPS verisi mevcut)
DATA_DIR            = "2026/THYZ_2026_Ornek_Veri_1.MP4"
CSV_PATH            = "2026/THYZ_2026_Ornek_Veri_1_translation.csv"
OUT_FIG             = "trajectory_comparison_2026_v1.png"
GPS_KESILME_KARESI  = 450
#
PROCESS_SCALE       = 0.5
# 2026 kamera kalibrasyonu (1920×1080 için, PROCESS_SCALE ile ölçekle):
FX = 1389.7 * PROCESS_SCALE
FY = 1387.1 * PROCESS_SCALE
CX = 954.007 * PROCESS_SCALE
CY = 558.896 * PROCESS_SCALE
# ─────────────────────────────────────────────────────────────────────────────

# GPS verisi olmadığında kullanılacak varsayılan değerler
DEFAULT_PPM         = 0.01   # m/px başlangıç tahmini
DEFAULT_HEADING_DEG = 90.0   # Kuzey = 90° (standart matematik açısı)

# Güvenilirlik eşiği: confidence bu değerin altına düşünce UNRELIABLE raporlanır
CONFIDENCE_ALARM = 0.4


def load_dataset(csv_path, data_dir):
    # CSV ayracını otomatik algıla (noktalı virgül veya virgül)
    with open(csv_path, "r") as f:
        first_line = f.readline()
    sep = ";" if ";" in first_line else ","
    df = pd.read_csv(csv_path, sep=sep)

    # Eksik kolonlar için varsayılan değer ekle
    if "translation_z" not in df.columns:
        df["translation_z"] = 0.0
    # gps_health_status yoksa gps_health_for_row GPS_KESILME_KARESI'ni kullanır

    # Video dosyası kontrolü
    is_video = False
    if os.path.isfile(data_dir) and data_dir.lower().endswith(('.mp4', '.avi', '.mov')):
        is_video = True
        print(f"Video dosyası algılandı: {data_dir}")
        return df, is_video, data_dir

    # Görüntü uzantısını otomatik algıla (.webp / .jpg / .png)
    sample = str(df.iloc[0]["frame_numbers"])
    img_ext = ".webp"
    for ext in [".webp", ".jpg", ".jpeg", ".png"]:
        if os.path.exists(os.path.join(data_dir, sample + ext)):
            img_ext = ext
            break
    print(f"Görüntü uzantısı: {img_ext}")

    df["frame_path"] = df["frame_numbers"].apply(
        lambda fn: os.path.join(data_dir, str(fn) + img_ext)
    )
    mask = df["frame_path"].apply(os.path.exists)
    missing = (~mask).sum()
    if missing:
        print(f"UYARI: {missing} kare dosyası bulunamadı, atlanıyor.")
        df = df[mask].reset_index(drop=True)
    return df, is_video, data_dir


def gps_health_for_row(row, frame_idx, has_health_col):
    """
    Satırdan GPS sağlık durumu döner.
    CSV'de gps_health_status varsa doğrudan okur.
    Yoksa GPS_KESILME_KARESI simülasyonunu kullanır.
    """
    if has_health_col:
        return int(row.get("gps_health_status", 1))
    if GPS_KESILME_KARESI < 0:
        return 1
    return 0 if frame_idx >= GPS_KESILME_KARESI else 1


def main():
    print("Hızlı test başlatılıyor...")

    K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]], dtype=np.float64)
    vo = MonocularVO()
    vo.K = K
    vo.update_calibration(focal=FX, pp=(CX, CY))
    print(f"Kalibrasyon — fx={FX:.1f}  fy={FY:.1f}  cx={CX:.1f}  cy={CY:.1f}")

    df, is_video, video_path = load_dataset(CSV_PATH, DATA_DIR)
    has_health_col = "gps_health_status" in df.columns

    # GPS referansı var mı? (GPS_KESILME_KARESI=0 veya veri tamamen sıfır)
    gps_all_zero = (
        float(df["translation_x"].max()) == 0.0 and
        float(df["translation_y"].max()) == 0.0
    )
    has_gps_reference = (GPS_KESILME_KARESI != 0) and not gps_all_zero

    print(f"Toplam kare: {len(df)} | GPS health kolonu: {'VAR' if has_health_col else 'YOK'}")
    if not has_gps_reference:
        print("  → GPS referansı yok: salt optik akış modu (PPM ve heading varsayılan)")
    elif not has_health_col:
        if GPS_KESILME_KARESI < 0:
            print("  → GPS hiç kesilmiyor (sadece GPS fazı testi)")
        else:
            print(f"  → Kare {GPS_KESILME_KARESI}'den itibaren GPS kesik simüle edilecek")

    gt_x, gt_y, gt_z = [], [], []
    pred_x, pred_y, pred_z = [], [], []
    conf_list          = []
    reliable_list      = []
    init_x, init_y, init_z = 0.0, 0.0, 0.0
    gps_cut_frame      = None   # grafik işareti için
    first_frame_done   = False

    start_time = time.time()
    prev_gps_status = 1

    cap = None
    if is_video:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"HATA: Video açılamadı: {video_path}")
            return

    for frame_idx, row in df.iterrows():
        if is_video:
            ret, frame = cap.read()
            if not ret:
                print("UYARI: Videonun sonuna gelindi veya kare okunamadı.")
                break
        else:
            frame = cv2.imread(row["frame_path"])
            if frame is None:
                print(f"UYARI: Kare okunamadı → {row['frame_path']}")
                continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if PROCESS_SCALE != 1.0:
            gray = cv2.resize(gray, None, fx=PROCESS_SCALE, fy=PROCESS_SCALE,
                              interpolation=cv2.INTER_AREA)

        true_x = float(row["translation_x"])
        true_y = float(row["translation_y"])
        true_z = float(row["translation_z"])

        if frame_idx == 0:
            init_x, init_y, init_z = true_x, true_y, true_z

        health = gps_health_for_row(row, frame_idx, has_health_col)

        if health == 1:
            json_data = {
                "translation_x": true_x,
                "translation_y": true_y,
                "translation_z": true_z,
                "gps_health_status": 1,
            }
            durum = "SAGLIKLI"
        else:
            json_data = {
                "translation_x": 0.0,
                "translation_y": 0.0,
                "translation_z": 0.0,
                "gps_health_status": 0,
            }
            durum = "KESILDI"
            if gps_cut_frame is None:
                gps_cut_frame = frame_idx

        # GPS kesilme anını logla
        if prev_gps_status == 1 and health == 0:
            if GPS_KESILME_KARESI != 0:
                print(f"\n*** GPS KESİLDİ — kare {frame_idx} ***\n")
        prev_gps_status = health

        sonuc = vo.process_frame(gray, json_data)

        # GPS referansı yoksa ilk kare sonrası varsayılan değerlerle başlat
        if not first_frame_done:
            first_frame_done = True
            if not vo.ppm_initialized or not vo.heading_initialized:
                vo.force_initialize(ppm=DEFAULT_PPM, heading_deg=DEFAULT_HEADING_DEG)
                print(f"  → VO varsayılan başlatıldı: PPM={DEFAULT_PPM}, Heading={DEFAULT_HEADING_DEG}°")
        px = sonuc["detected_translations"][0]["translation_x"]
        py = sonuc["detected_translations"][0]["translation_y"]
        pz = sonuc["detected_translations"][0]["translation_z"]
        conf = sonuc["confidence"]
        reliable = sonuc["is_reliable"]

        gt_x.append(true_x - init_x)
        gt_y.append(true_y - init_y)
        gt_z.append(true_z - init_z)
        pred_x.append(px)
        pred_y.append(py)
        pred_z.append(pz)
        conf_list.append(conf)
        reliable_list.append(reliable)

        # Güvenilirlik alarmı
        if not reliable and durum == "KESILDI":
            if frame_idx % 50 == 0:
                print(f"  [ALARM] Kare {frame_idx:05d} — UNRELIABLE | "
                      f"conf={conf:.2f} | inlier={sonuc['inlier_ratio']:.2f} | "
                      f"feat={sonuc['feature_count']}")

        if frame_idx % 200 == 0:
            elapsed = time.time() - start_time
            fps = (frame_idx + 1) / max(elapsed, 1e-6)
            h_ok = f"H={sonuc['heading_deg']}°" if sonuc['heading_deg'] is not None else "H=?"
            print(
                f"Kare {frame_idx:05d} | GPS: {durum} | "
                f"GT: ({true_x-init_x:.2f}, {true_y-init_y:.2f}) "
                f"-> VO: ({px:.2f}, {py:.2f}) | "
                f"conf={conf:.2f} | PPM={vo.pixel_to_meter:.5f} | {h_ok} | {fps:.1f} FPS"
            )

    # ── Metrikler ────────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    n = len(gt_x)
    print(f"\nToplam {n} kare, {elapsed:.1f}s ({n/max(elapsed,1e-6):.1f} FPS)")

    pred_arr_2d = np.array(list(zip(pred_x, pred_y)))
    gt_arr_2d   = np.array(list(zip(gt_x, gt_y)))
    pred_arr_3d = np.array(list(zip(pred_x, pred_y, pred_z)))
    gt_arr_3d   = np.array(list(zip(gt_x, gt_y, gt_z)))
    cut = gps_cut_frame if gps_cut_frame is not None else n

    if has_gps_reference:
        errors_2d = np.linalg.norm(gt_arr_2d - pred_arr_2d, axis=1)
        errors_3d = np.linalg.norm(gt_arr_3d - pred_arr_3d, axis=1)
        errors_z  = np.abs(gt_arr_3d[:, 2] - pred_arr_3d[:, 2])
        
        rmse_gps_3d = np.sqrt(np.mean(errors_3d[:cut]**2)) if cut > 0 else float("nan")
        rmse_dr_3d  = np.sqrt(np.mean(errors_3d[cut:]**2)) if cut < n else float("nan")
        
        rmse_2d = np.sqrt(np.mean(errors_2d**2))
        rmse_3d = np.sqrt(np.mean(errors_3d**2))
        rmse_z  = np.sqrt(np.mean(errors_z**2))
        
        unreliable_dr = sum(1 for i, r in enumerate(reliable_list) if i >= cut and not r)
        print(f"\n=== HATA METRİKLERİ ===")
        print(f"Toplam 2D RMSE       : {rmse_2d:.3f} m")
        print(f"Toplam Z RMSE        : {rmse_z:.3f} m")
        print(f"Toplam 3D RMSE       : {rmse_3d:.3f} m")
        print(f"GPS-sağlıklı 3D RMSE : {rmse_gps_3d:.3f} m  (kare 0–{cut})")
        print(f"Dead-reckoning 3D RMSE: {rmse_dr_3d:.3f} m  (kare {cut}–{n})")
        print(f"Max 3D hata          : {errors_3d.max():.3f} m  (kare {int(errors_3d.argmax())})")
        print(f"Unreliable DR kareler: {unreliable_dr} / {n - cut}")
        print("========================")
    else:
        errors = np.zeros(n)  # GT yok, hata hesaplanamaz
        print(f"\n=== OPTİK AKIŞ ÖZET ===")
        print(f"GPS referansı olmadığından hata hesaplanamaz.")
        print(f"VO yörünge ucu: ({pred_x[-1]:.2f}, {pred_y[-1]:.2f}) m  (ölçek tahmini)")
        print("========================")

    if vo.heading_initialized:
        print(f"Son heading          : {math.degrees(vo.heading_angle):.1f}°  (0=Doğu, 90=Kuzey)")

    # ── Grafik ──────────────────────────────────────────────────────────────
    print("Grafik çiziliyor...")
    fig = plt.figure(figsize=(18, 18))
    gs = fig.add_gridspec(3, 6)

    # Sol Üst: Yörünge
    ax = fig.add_subplot(gs[0, :3])
    if has_gps_reference:
        ax.plot(gt_x, gt_y, label="Gerçek Rota (GT)", color="blue", linewidth=2)

    # VO yörüngesi: confidence'a göre renkli
    if n > 1:
        pts = np.array(list(zip(pred_x, pred_y))).reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        colors_norm = np.array(conf_list[:-1])
        lc = LineCollection(segs, cmap="RdYlGn", norm=plt.Normalize(0, 1))
        lc.set_array(colors_norm)
        lc.set_linewidth(2)
        ax.add_collection(lc)
        fig.colorbar(lc, ax=ax, label="Confidence (0=kötü, 1=iyi)")

    if has_gps_reference and gps_cut_frame is not None and gps_cut_frame < len(gt_x):
        ax.scatter(gt_x[gps_cut_frame], gt_y[gps_cut_frame],
                   color="green", s=150, zorder=5,
                   label=f"GPS Kesintisi (Kare {gps_cut_frame})")

    title = "VO Yörüngesi (GPS referansı yok)" if not has_gps_reference else "Yörünge Karşılaştırması"
    ax.set_title(title)
    ax.set_xlabel("X (tahmini m)")
    ax.set_ylabel("Y (tahmini m)")
    if has_gps_reference:
        ax.legend()
    ax.grid(True)
    ax.axis("equal")

    # Sağ Üst: Confidence zaman serisi (+ hata varsa)
    ax2 = fig.add_subplot(gs[0, 3:])
    frames = list(range(n))
    if has_gps_reference:
        ax2.plot(frames, errors_3d, color="red", linewidth=1, alpha=0.7, label="Konum Hatası (3D) (m)")
        ax2.set_ylabel("Hata (m)", color="red")
        ax2.tick_params(axis="y", labelcolor="red")

    ax3 = ax2.twinx() if has_gps_reference else ax2
    ax3.plot(frames, conf_list, color="green", linewidth=1, alpha=0.7, label="Confidence")
    ax3.axhline(CONFIDENCE_ALARM, color="orange", linestyle="--", linewidth=1, label=f"Alarm eşiği ({CONFIDENCE_ALARM})")
    ax3.set_ylabel("Confidence", color="green")
    ax3.tick_params(axis="y", labelcolor="green")
    ax3.set_ylim(0, 1.1)

    if gps_cut_frame is not None:
        ax2.axvline(gps_cut_frame, color="gray", linestyle=":", linewidth=1.5, label="GPS Kesildi")

    ax2.set_xlabel("Kare")
    title2 = "Güvenilirlik Zaman Serisi" if not has_gps_reference else "Hata & Güvenilirlik Zaman Serisi"
    ax2.set_title(title2)
    ax2.grid(True, alpha=0.3)

    if has_gps_reference:
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines3, labels3 = ax3.get_legend_handles_labels()
        ax2.legend(lines2 + lines3, labels2 + labels3, loc="upper left", fontsize=8)
    else:
        ax2.legend(loc="upper right", fontsize=8)

    # Orta Satır: X, Y, Z Grafikleri
    ax_x = fig.add_subplot(gs[1, 0:2])
    ax_y = fig.add_subplot(gs[1, 2:4])
    ax_z = fig.add_subplot(gs[1, 4:6])

    for a, gt_val, pred_val, label in zip([ax_x, ax_y, ax_z], 
                                          [gt_x, gt_y, gt_z], 
                                          [pred_x, pred_y, pred_z], 
                                          ["X Ekseni (m)", "Y Ekseni (m)", "Z Ekseni (m)"]):
        if has_gps_reference:
            a.plot(frames, gt_val, label="GT", color="blue", linewidth=1.5, alpha=0.8)
        a.plot(frames, pred_val, label="VO", color="red", linewidth=1.5, alpha=0.8, linestyle="--")
        
        if gps_cut_frame is not None:
            a.axvline(gps_cut_frame, color="green", linestyle=":", linewidth=2, label="GPS Kesildi")
            
        a.set_xlabel("Kare")
        a.set_ylabel(label)
        a.set_title(f"Zamana Göre {label} Değişimi")
        a.grid(True, alpha=0.3)
        a.legend(fontsize=8)
        
    # Alt Satır: 3 Boyutlu Grafikler
    ax_3d = fig.add_subplot(gs[2, 1:5], projection='3d')
    if has_gps_reference:
        ax_3d.plot(gt_x, gt_y, gt_z, label="GT (Gerçek Rota)", color="blue", linewidth=2, alpha=0.8)
        
    ax_3d.plot(pred_x, pred_y, pred_z, label="VO (Tahmin)", color="red", linewidth=2, alpha=0.8, linestyle="--")
    
    if has_gps_reference and gps_cut_frame is not None and gps_cut_frame < len(gt_x):
        ax_3d.scatter(gt_x[gps_cut_frame], gt_y[gps_cut_frame], gt_z[gps_cut_frame],
                      color="green", s=100, zorder=5, label="GPS Kesintisi")
                      
    ax_3d.set_xlabel("X (m)")
    ax_3d.set_ylabel("Y (m)")
    ax_3d.set_zlabel("Z (m)")
    ax_3d.set_title("3 Boyutlu Yörünge Karşılaştırması")
    ax_3d.legend()

    if cap is not None:
        cap.release()

    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=150)
    print(f"Grafik '{OUT_FIG}' olarak kaydedildi.")
    import subprocess, sys
    subprocess.Popen(["open", OUT_FIG])


if __name__ == "__main__":
    main()
