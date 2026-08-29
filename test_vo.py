import cv2
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mono_vo import MonocularVO

# ── Veri Kaynağı Yapılandırması ──────────────────────────────────────────────
DATA_DIR = "2025/THYZ_2025_Oturum_2-2"
CSV_PATH = "2025/THYZ_2025_Oturum_2_Translation.csv"
OUT_FIG  = "trajectory_comparison.png"

GPS_KESILME_KARESI = 300

# İşlem boyutu: 0.5 = yarı çözünürlük (4x hız), 1.0 = tam 4K
PROCESS_SCALE = 0.5

# RGB 4K kamera kalibrasyonu (ölçeklendirilmiş)
SCALE_W = (3840 / 4000) * PROCESS_SCALE
SCALE_H = (2160 / 3000) * PROCESS_SCALE
FX = 2792.2 * SCALE_W
FY = 2795.2 * SCALE_H
CX = 1988.0 * SCALE_W
CY = 1562.2 * SCALE_H
# ─────────────────────────────────────────────────────────────────────────────


def load_dataset(csv_path, data_dir):
    df = pd.read_csv(csv_path)
    df["frame_path"] = df["frame_numbers"].apply(
        lambda fn: os.path.join(data_dir, fn + ".webp")
    )
    mask = df["frame_path"].apply(os.path.exists)
    missing = (~mask).sum()
    if missing:
        print(f"UYARI: {missing} kare dosyası bulunamadı, atlanıyor.")
        df = df[mask].reset_index(drop=True)
    return df


def main():
    print("Test başlatılıyor...")

    K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]], dtype=np.float64)
    vo = MonocularVO()
    vo.K = K
    vo.update_calibration(focal=FX, pp=(CX, CY))

    print(f"CSV yükleniyor: {CSV_PATH}")
    df = load_dataset(CSV_PATH, DATA_DIR)
    print(f"Toplam kare: {len(df)}")

    init_x = float(df.iloc[0]["translation_x"])
    init_y = float(df.iloc[0]["translation_y"])

    gt_x,   gt_y   = [], []
    pred_x, pred_y = [], []

    for frame_idx, row in df.iterrows():
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

        if frame_idx >= GPS_KESILME_KARESI:
            json_data = {
                "translation_x": 0.0,
                "translation_y": 0.0,
                "translation_z": 0.0,
                "gps_health_status": 0,
            }
            durum = "KESILDI"
        else:
            json_data = {
                "translation_x": true_x,
                "translation_y": true_y,
                "translation_z": true_z,
                "gps_health_status": 1,
            }
            durum = "SAGLIKLI"

        sonuc = vo.process_frame(gray, json_data)
        px = sonuc["detected_translations"][0]["translation_x"]
        py = sonuc["detected_translations"][0]["translation_y"]

        gt_rx = true_x - init_x
        gt_ry = true_y - init_y

        gt_x.append(gt_rx)
        gt_y.append(gt_ry)
        pred_x.append(px)
        pred_y.append(py)

        if frame_idx % 50 == 0:
            print(
                f"Kare {frame_idx:04d} | GPS: {durum} "
                f"| GT: ({gt_rx:.2f}, {gt_ry:.2f}) "
                f"-> VO: ({px:.2f}, {py:.2f})"
            )

        # ── Görsel Gösterim ──────────────────────────────────────────────────
        # İzlenen noktaları işaretle
        if vo.px_ref is not None and len(vo.px_ref) > 0:
            for pt in vo.px_ref:
                cv2.circle(frame, (int(pt[0]), int(pt[1])), 3, (0, 255, 0), -1)

        renk = (0, 255, 0) if durum == "SAGLIKLI" else (0, 0, 255)
        cv2.putText(frame, f"Kare: {frame_idx} | GPS: {durum}",
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, renk, 2)
        cv2.putText(frame, f"Hata(X): {abs(gt_rx - px):.2f}m",
                    (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(frame, f"Hata(Y): {abs(gt_ry - py):.2f}m",
                    (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

        frame_small = cv2.resize(frame, (1280, 720))
        cv2.imshow("Monocular VO Test — [Q] cikis", frame_small)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Kullanıcı tarafından durduruldu.")
            break

    cv2.destroyAllWindows()

    # ── Hata Metrikleri ──────────────────────────────────────────────────────
    n = len(gt_x)
    if n > 0:
        gt_arr   = np.array(list(zip(gt_x, gt_y)))
        pred_arr = np.array(list(zip(pred_x, pred_y)))
        errors   = np.linalg.norm(gt_arr - pred_arr, axis=1)

        gps_cut = min(GPS_KESILME_KARESI, n)
        print(f"\n=== HATA METRİKLERİ ===")
        print(f"Toplam RMSE          : {np.sqrt(np.mean(errors**2)):.3f} m")
        print(f"GPS-sağlıklı RMSE    : {np.sqrt(np.mean(errors[:gps_cut]**2)):.3f} m")
        print(f"Dead-reckoning RMSE  : {np.sqrt(np.mean(errors[gps_cut:]**2)):.3f} m")
        print(f"Max hata             : {errors.max():.3f} m  (kare {int(errors.argmax())})")
        print("========================")

    # ── Grafik ───────────────────────────────────────────────────────────────
    print("Grafik çiziliyor...")
    plt.figure(figsize=(10, 8))
    plt.plot(gt_x,   gt_y,   label="Gerçek Rota (Ground Truth)", color="blue", linewidth=2)
    plt.plot(pred_x, pred_y, label="VO Tahmini", color="red", linestyle="--", linewidth=2)

    if n > GPS_KESILME_KARESI:
        plt.scatter(
            gt_x[GPS_KESILME_KARESI], gt_y[GPS_KESILME_KARESI],
            color="green", s=150,
            label=f"GPS Kesintisi (Kare {GPS_KESILME_KARESI})",
            zorder=5,
        )

    plt.title("İHA Yörünge Karşılaştırması (Gerçek vs VO)")
    plt.xlabel("X Ekseni (m)")
    plt.ylabel("Y Ekseni (m)")
    plt.legend()
    plt.grid(True)
    plt.savefig(OUT_FIG, dpi=150)
    print(f"Test tamamlandı. Grafik '{OUT_FIG}' olarak kaydedildi.")


if __name__ == "__main__":
    main()
