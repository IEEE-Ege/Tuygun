"""Hazır kare klasörünü frame-by-frame işleyen çalıştırıcı.

Video sistemiyle aynı VO çekirdeğini (vo/) kullanır; fark, girdinin video
yerine diskteki kare dosyaları olması ve CSV'nin opsiyonel olmasıdır.
"""
import os
import re
import sys
import time
import glob
import dataclasses

import cv2
import pandas as pd

# Proje kökünü path'e ekle ve çalışma dizini yap (veri yolları köke göredir)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

from vo.sensor_configs import get_config
from vo.vo_pipeline import VOPipeline
from vo.dataset_loader import gps_health_for_row
from vo.reporting import compute_metrics, print_optical_flow_summary, save_csv, plot_trajectory
from frame_sistemi.frame_ayarlar import DEFAULT_PPM, DEFAULT_HEADING_DEG


def _dogal_sirala(yollar):
    """Dosya adındaki sayıya göre sıralar (frame_2 < frame_10)."""
    def anahtar(yol):
        sayilar = re.findall(r"\d+", os.path.basename(yol))
        return int(sayilar[-1]) if sayilar else 0
    return sorted(yollar, key=anahtar)


GECERLI_UZANTILAR = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")


def _uzanti_bul(frame_dir):
    """Klasördeki en yaygın görüntü uzantısını otomatik bulur."""
    sayilar = {}
    for dosya in os.listdir(frame_dir):
        ext = os.path.splitext(dosya)[1].lower()
        if ext in GECERLI_UZANTILAR:
            sayilar[ext] = sayilar.get(ext, 0) + 1
    if not sayilar:
        return None
    return max(sayilar, key=sayilar.get)


def _csv_bul(frame_dir, varsayilan):
    """Önce kare klasörünün içinde CSV arar; yoksa ayarlardaki yolu kullanır."""
    icerdekiler = sorted(glob.glob(os.path.join(frame_dir, "*.csv")))
    if icerdekiler:
        return icerdekiler[0]
    return varsayilan


def calistir(sensor_type, ayar):
    print(f"Frame-by-frame sistem başlatılıyor — sensör: {sensor_type}")

    frame_dir = ayar["frame_dir"]
    if not os.path.isdir(frame_dir):
        print(f"HATA: Kare klasörü bulunamadı: {frame_dir}")
        print(f"  → Kareleri şu klasöre koyun: {frame_dir}/")
        return

    # Uzantı: ayarlarda None ise otomatik algıla
    uzanti = ayar.get("uzanti") or _uzanti_bul(frame_dir)
    if uzanti is None:
        print(f"HATA: {frame_dir} içinde görüntü dosyası yok "
              f"(desteklenen: {', '.join(GECERLI_UZANTILAR)})")
        return

    kareler = _dogal_sirala(glob.glob(os.path.join(frame_dir, f"*{uzanti}")))
    if not kareler:
        print(f"HATA: {frame_dir} içinde '{uzanti}' uzantılı kare yok.")
        return
    print(f"{len(kareler)} kare bulundu (uzantı: {uzanti})")

    # Kalibrasyon ve VO'yu video sistemiyle aynı profilden kur
    cfg = get_config(sensor_type)
    cfg = dataclasses.replace(cfg, gps_kesilme_karesi=ayar["gps_kesilme_karesi"],
                              out_fig=ayar["out_fig"])
    pipeline = VOPipeline(cfg)
    vo = pipeline.vo

    # CSV: önce kare klasörünün İÇİNDE ara (kareyle birlikte verilmişse),
    # yoksa ayarlardaki yolu kullan. Hiçbiri yoksa salt optik akış modu.
    csv_path = _csv_bul(frame_dir, ayar["csv_path"])
    df = None
    if csv_path is not None:
        if not os.path.exists(csv_path):
            print(f"UYARI: CSV bulunamadı ({csv_path}) — salt optik akış moduna geçiliyor.")
            csv_path = None
    if csv_path is not None:
        print(f"CSV kullanılıyor: {csv_path}")
        with open(csv_path) as f:
            sep = ";" if ";" in f.readline() else ","
        df = pd.read_csv(csv_path, sep=sep)
        if "translation_z" not in df.columns:
            df["translation_z"] = 0.0
        # Kare dosya adı → CSV satırı eşlemesi (frame_numbers üzerinden)
        df["_key"] = df["frame_numbers"].astype(str)
        satirlar = {k: r for k, r in zip(df["_key"], df.to_dict("records"))}
        print(f"CSV yüklendi: {len(df)} satır")
    has_health_col = df is not None and "gps_health_status" in df.columns

    gt_x, gt_y, gt_z = [], [], []
    pred_x, pred_y, pred_z = [], [], []
    conf_list, reliable_list = [], []
    init_x = init_y = init_z = 0.0
    gps_cut_frame = None
    first_frame_done = False
    start_time = time.time()

    for frame_idx, yol in enumerate(kareler):
        frame = cv2.imread(yol)
        if frame is None:
            print(f"UYARI: Kare okunamadı → {yol}")
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if cfg.process_scale != 1.0:
            gray = cv2.resize(gray, None, fx=cfg.process_scale, fy=cfg.process_scale,
                              interpolation=cv2.INTER_AREA)

        pipeline.apply_frame_calibration(gray)

        # GT satırını dosya adından bul (uzantısız ad = frame_numbers)
        row = None
        if df is not None:
            ad = os.path.splitext(os.path.basename(yol))[0]
            row = satirlar.get(ad)
            if row is None and frame_idx < len(df):
                row = df.iloc[frame_idx].to_dict()  # ad eşleşmezse sıra ile

        if row is not None:
            true_x = float(row["translation_x"])
            true_y = float(row["translation_y"])
            true_z = float(row["translation_z"])
            health = gps_health_for_row(row, frame_idx, has_health_col,
                                        cfg.gps_kesilme_karesi)
        else:
            true_x = true_y = true_z = 0.0
            health = 0  # CSV yok: GPS hiç yok, salt optik akış

        if frame_idx == 0:
            init_x, init_y, init_z = true_x, true_y, true_z

        if health == 1:
            json_data = {"translation_x": true_x, "translation_y": true_y,
                         "translation_z": true_z, "gps_health_status": 1}
        else:
            json_data = {"translation_x": 0.0, "translation_y": 0.0,
                         "translation_z": 0.0, "gps_health_status": 0}
            if gps_cut_frame is None:
                gps_cut_frame = frame_idx

        sonuc = vo.process_frame(gray, json_data)

        if not first_frame_done:
            first_frame_done = True
            if not vo.ppm_initialized or not vo.heading_initialized:
                vo.force_initialize(ppm=DEFAULT_PPM, heading_deg=DEFAULT_HEADING_DEG)
                print(f"  → VO varsayılan başlatıldı: PPM={DEFAULT_PPM}, "
                      f"Heading={DEFAULT_HEADING_DEG}°")

        t = sonuc["detected_translations"][0]
        gt_x.append(true_x - init_x); gt_y.append(true_y - init_y); gt_z.append(true_z - init_z)
        pred_x.append(t["translation_x"]); pred_y.append(t["translation_y"]); pred_z.append(t["translation_z"])
        conf_list.append(sonuc["confidence"])
        reliable_list.append(sonuc["is_reliable"])

        if frame_idx % 200 == 0:
            fps = (frame_idx + 1) / max(time.time() - start_time, 1e-6)
            print(f"Kare {frame_idx:05d} | GPS: {'SAGLIKLI' if health else 'KESILDI'} | "
                  f"VO: ({t['translation_x']:.2f}, {t['translation_y']:.2f}) | "
                  f"conf={sonuc['confidence']:.2f} | {fps:.1f} FPS")

    elapsed = time.time() - start_time
    has_gps_reference = df is not None and cfg.gps_kesilme_karesi != 0

    result = {
        "gt_x": gt_x, "gt_y": gt_y, "gt_z": gt_z,
        "pred_x": pred_x, "pred_y": pred_y, "pred_z": pred_z,
        "conf_list": conf_list, "reliable_list": reliable_list,
        "gps_cut_frame": gps_cut_frame,
        "has_gps_reference": has_gps_reference,
        "elapsed": elapsed, "vo": vo,
    }

    save_csv(result, "frame_sistemi/cikti/trajectory_data.csv")
    metrics = None
    if has_gps_reference:
        metrics = compute_metrics(result)
    else:
        print_optical_flow_summary(result)

    plot_trajectory(result, metrics, cfg.out_fig,
                    confidence_alarm=cfg.confidence_alarm, open_after=True)
