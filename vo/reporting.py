import csv
import math
import subprocess

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def compute_metrics(result):
    gt_x, gt_y, gt_z = result["gt_x"], result["gt_y"], result["gt_z"]
    pred_x, pred_y, pred_z = result["pred_x"], result["pred_y"], result["pred_z"]
    n = len(gt_x)
    cut = result["gps_cut_frame"] if result["gps_cut_frame"] is not None else n

    pred_arr_2d = np.array(list(zip(pred_x, pred_y)))
    gt_arr_2d = np.array(list(zip(gt_x, gt_y)))
    pred_arr_3d = np.array(list(zip(pred_x, pred_y, pred_z)))
    gt_arr_3d = np.array(list(zip(gt_x, gt_y, gt_z)))

    errors_2d = np.linalg.norm(gt_arr_2d - pred_arr_2d, axis=1)
    errors_3d = np.linalg.norm(gt_arr_3d - pred_arr_3d, axis=1)
    errors_z = np.abs(gt_arr_3d[:, 2] - pred_arr_3d[:, 2])

    rmse_gps_3d = np.sqrt(np.mean(errors_3d[:cut] ** 2)) if cut > 0 else float("nan")
    rmse_dr_3d = np.sqrt(np.mean(errors_3d[cut:] ** 2)) if cut < n else float("nan")

    rmse_2d = np.sqrt(np.mean(errors_2d ** 2))
    rmse_3d = np.sqrt(np.mean(errors_3d ** 2))
    rmse_z = np.sqrt(np.mean(errors_z ** 2))

    reliable_list = result["reliable_list"]
    unreliable_dr = sum(1 for i, r in enumerate(reliable_list) if i >= cut and not r)

    metrics = {
        "n": n, "cut": cut,
        "errors_2d": errors_2d, "errors_3d": errors_3d, "errors_z": errors_z,
        "rmse_2d": rmse_2d, "rmse_3d": rmse_3d, "rmse_z": rmse_z,
        "rmse_gps_3d": rmse_gps_3d, "rmse_dr_3d": rmse_dr_3d,
        "unreliable_dr": unreliable_dr,
    }

    print(f"\nToplam {n} kare, {result['elapsed']:.1f}s ({n/max(result['elapsed'],1e-6):.1f} FPS)")
    print(f"\n=== HATA METRİKLERİ ===")
    print(f"Toplam 2D RMSE       : {rmse_2d:.3f} m")
    print(f"Toplam Z RMSE        : {rmse_z:.3f} m")
    print(f"Toplam 3D RMSE       : {rmse_3d:.3f} m")
    print(f"GPS-sağlıklı 3D RMSE : {rmse_gps_3d:.3f} m  (kare 0–{cut})")
    print(f"Dead-reckoning 3D RMSE: {rmse_dr_3d:.3f} m  (kare {cut}–{n})")
    print(f"Max 3D hata          : {errors_3d.max():.3f} m  (kare {int(errors_3d.argmax())})")
    print(f"Unreliable DR kareler: {unreliable_dr} / {n - cut}")
    print("========================")

    vo = result["vo"]
    if vo.heading_initialized:
        print(f"Son heading          : {math.degrees(vo.heading_angle):.1f}°  (0=Doğu, 90=Kuzey)")

    return metrics


def print_optical_flow_summary(result):
    pred_x, pred_y = result["pred_x"], result["pred_y"]
    print(f"\n=== OPTİK AKIŞ ÖZET ===")
    print(f"GPS referansı olmadığından hata hesaplanamaz.")
    print(f"VO yörünge ucu: ({pred_x[-1]:.2f}, {pred_y[-1]:.2f}) m  (ölçek tahmini)")
    print("========================")


def save_csv(result, path="cikti/trajectory_data.csv"):
    print(f"Yörünge verileri {path} olarak kaydediliyor...")
    gt_x, gt_y, gt_z = result["gt_x"], result["gt_y"], result["gt_z"]
    pred_x, pred_y, pred_z = result["pred_x"], result["pred_y"], result["pred_z"]
    conf_list = result["conf_list"]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "gt_x", "gt_y", "gt_z", "pred_x", "pred_y", "pred_z", "conf"])
        for i in range(len(gt_x)):
            writer.writerow([i, gt_x[i], gt_y[i], gt_z[i], pred_x[i], pred_y[i], pred_z[i], conf_list[i]])


def plot_trajectory(result, metrics, out_fig, confidence_alarm=0.4, open_after=True):
    gt_x, gt_y, gt_z = result["gt_x"], result["gt_y"], result["gt_z"]
    pred_x, pred_y, pred_z = result["pred_x"], result["pred_y"], result["pred_z"]
    conf_list = result["conf_list"]
    has_gps_reference = result["has_gps_reference"]
    gps_cut_frame = result["gps_cut_frame"]
    n = metrics["n"] if metrics is not None else len(gt_x)

    print("Grafik çiziliyor...")
    fig = plt.figure(figsize=(18, 18))
    gs = fig.add_gridspec(3, 6)

    # Sol Üst: Yörünge
    ax = fig.add_subplot(gs[0, :3])
    if has_gps_reference:
        ax.plot(gt_x, gt_y, label="Gerçek Rota (GT)", color="blue", linewidth=2)

    if n > 1:
        pts = np.array(list(zip(pred_x, pred_y))).reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        colors_norm = np.array(conf_list[:-1])
        lc = LineCollection(segs, cmap="RdYlGn", norm=plt.Normalize(0, 1))
        lc.set_array(colors_norm)
        lc.set_linewidth(1)
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
    all_x = np.concatenate([gt_x, pred_x])
    all_y = np.concatenate([gt_y, pred_y])
    margin = 30
    ax.set_xlim([np.min(all_x) - margin, np.max(all_x) + margin])
    ax.set_ylim([np.min(all_y) - margin, np.max(all_y) + margin])
    ax.set_aspect('equal', adjustable='box')

    if has_gps_reference:
        metrics_text = (
            f"Hata Metrikleri:\n"
            f"2D RMSE : {metrics['rmse_2d']:.2f} m\n"
            f"Z RMSE  : {metrics['rmse_z']:.2f} m\n"
            f"3D RMSE : {metrics['rmse_3d']:.2f} m\n"
            f"Max Hata: {metrics['errors_3d'].max():.2f} m"
        )
        fig.text(0.02, 0.50, metrics_text, fontsize=12, va='center', ha='left',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='gray', alpha=0.9))

    # Sağ Üst: Confidence zaman serisi (+ hata varsa)
    ax2 = fig.add_subplot(gs[0, 3:])
    frames = list(range(n))
    if has_gps_reference:
        ax2.plot(frames, metrics["errors_3d"], color="red", linewidth=1, alpha=0.7, label="Konum Hatası (3D) (m)")
        ax2.set_ylabel("Hata (m)", color="red")
        ax2.tick_params(axis="y", labelcolor="red")

    ax3 = ax2.twinx() if has_gps_reference else ax2
    ax3.plot(frames, conf_list, color="green", linewidth=1, alpha=0.7, label="Confidence")
    ax3.axhline(confidence_alarm, color="orange", linestyle="--", linewidth=1, label=f"Alarm eşiği ({confidence_alarm})")
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
        ax2.legend(handles=lines2 + lines3, labels=labels2 + labels3, loc="upper right")
    else:
        ax2.legend(loc="upper right", fontsize=8)

    # 3D Yörünge Grafiği (Orta Sağ)
    ax4 = fig.add_subplot(gs[1, :], projection='3d')
    if has_gps_reference:
        ax4.plot(gt_x, gt_y, gt_z, label="Gerçek Rota (GT)", color="blue", linewidth=2)

    if n > 1:
        sc = ax4.scatter(pred_x, pred_y, pred_z, c=conf_list, cmap="RdYlGn", vmin=0, vmax=1, s=2, alpha=0.8, label="Tahmini Rota (VO)")
        ax4.plot(pred_x, pred_y, pred_z, color="gray", linewidth=0.5, alpha=0.3)
        fig.colorbar(sc, ax=ax4, label="Confidence", pad=0.1)

    if has_gps_reference and gps_cut_frame is not None and gps_cut_frame < len(gt_x):
        ax4.scatter(gt_x[gps_cut_frame], gt_y[gps_cut_frame], gt_z[gps_cut_frame],
                    color="green", s=150, zorder=5, label="GPS Kesintisi")

    ax4.set_title("3D Yörünge Karşılaştırması")
    ax4.set_xlabel("X (m)")
    ax4.set_ylabel("Y (m)")
    ax4.set_zlabel("Z (m)")

    # Eksenleri eşit ölçeklendir (en çok sapma yapan eksene göre)
    all_x = np.concatenate([gt_x, pred_x])
    all_y = np.concatenate([gt_y, pred_y])
    all_z = np.concatenate([gt_z, pred_z])
    max_range = np.array([all_x.max() - all_x.min(), all_y.max() - all_y.min(), all_z.max() - all_z.min()]).max() / 2.0
    mid_x = (all_x.max() + all_x.min()) * 0.5
    mid_y = (all_y.max() + all_y.min()) * 0.5
    mid_z = (all_z.max() + all_z.min()) * 0.5

    ax4.set_xlim(mid_x - max_range, mid_x + max_range)
    ax4.set_ylim(mid_y - max_range, mid_y + max_range)
    ax4.set_zlim(mid_z - max_range, mid_z + max_range)
    if has_gps_reference:
        ax4.legend()

    # Orta Satır: X, Y, Z Grafikleri
    ax_x = fig.add_subplot(gs[2, 0:2])
    ax_y = fig.add_subplot(gs[2, 2:4])
    ax_z = fig.add_subplot(gs[2, 4:6])

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

    plt.tight_layout()
    plt.savefig(out_fig, dpi=150)
    print(f"Grafik '{out_fig}' olarak kaydedildi.")
    if open_after:
        subprocess.Popen(["open", out_fig])
