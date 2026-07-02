# ═══════════════════════════════════════════════════════════════════════════
#  TOPLU TEST — tüm yılların verisini tek komutla koşar, tek tablo basar
#
#  Amaç: "her duruma hazır" tek bir genel model. Parametre değişikliği
#  yapıldığında bu script koşulur; hiçbir veri setinde gerileme olmamalı.
#
#      python3 araclar/batch_test.py            # tüm senaryolar
#      python3 araclar/batch_test.py 2025       # adında '2025' geçenler
# ═══════════════════════════════════════════════════════════════════════════
import dataclasses
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

from vo.sensor_configs import get_config
from vo.vo_pipeline import VOPipeline
from vo.reporting import compute_metrics

# (etiket, profil, geçersiz kılmalar)
# Not: 2025 Termal uçuşunda ilk ~500 kare hover — kare 450 kesintisiyle ölçek
# öğrenilemez (fiziksel imkânsız). Bu yüzden o veri için kesinti 1500'de.
SENARYOLAR = [
    ("2024 RGB",            "rgb", dict(
        data_dir="yillar/2024/TUYZ_2024_Ornek_Video.MP4",
        csv_path="yillar/2024/GT_Translations.csv",
        fx=1413.3, fy=1418.8, cx=950.0639, cy=543.3796,
        calib_width=1920, calib_height=1080,
    )),
    ("2026 RGB",            "rgb", dict(
        data_dir="yillar/2026/THYZ_2026_Ornek_Veri_1.MP4",
        csv_path="yillar/2026/THYZ_2026_Ornek_Veri_1_translation.csv",
        fx=1389.7, fy=1387.1, cx=954.007, cy=558.896,
        calib_width=1920, calib_height=1080,
    )),
    ("2026 Termal",         "thermal", dict(
        data_dir="yillar/2026/THYZ_2026_Ornek_Veri_2_Termal.MP4",
        csv_path="yillar/2026/THYZ_2026_Ornek_Veri_2_Termal_translation.csv",
    )),
    ("2025 RGB Veri-2",     "rgb_2025", dict(
        data_dir="yillar/2025/Ornek-Veri-2-RGB.MP4",
        csv_path="yillar/2025/Ornek-Veri-2-RGB-translation.csv",
    )),
    ("2025 RGB Veri-1",     "rgb_2025", dict(
        data_dir="yillar/2025/Ornek-Veri-1-RGB.MP4",
        csv_path="yillar/2025/Ornek-Veri-1-RGB-translation.csv",
    )),
    ("2025 Termal Veri-2",  "thermal_2025", dict(gps_kesilme_karesi=1500)),
    ("2025 Termal Veri-1",  "thermal_2025", dict(
        data_dir="yillar/2025/Ornek-Veri-1-Termal.MP4",
        csv_path="yillar/2025/Ornek-Veri-1-Termal-translation.csv",
    )),
]


def main():
    filtre = sys.argv[1].lower() if len(sys.argv) > 1 else ""
    sonuclar = []

    for etiket, profil, overrides in SENARYOLAR:
        if filtre and filtre not in etiket.lower():
            continue
        cfg = get_config(profil)
        overrides = dict(overrides)
        overrides["out_fig"] = (
            "video_sistemi/cikti/batch_"
            + etiket.lower().replace(" ", "_").replace("-", "") + ".png")
        cfg = dataclasses.replace(cfg, **overrides)

        if not os.path.exists(cfg.data_dir) or not os.path.exists(cfg.csv_path):
            print(f"\n─── {etiket}: VERİ YOK, atlanıyor ({cfg.data_dir}) ───")
            continue

        print(f"\n{'═'*70}\n  {etiket}  (profil: {profil}, kesinti: kare {cfg.gps_kesilme_karesi})\n{'═'*70}")
        result = VOPipeline(cfg).run()
        if result is None:
            continue
        m = compute_metrics(result)
        sonuclar.append((etiket, m))

    if not sonuclar:
        return
    print(f"\n\n{'═'*70}\n  ÖZET TABLO\n{'═'*70}")
    print(f"{'Veri seti':<22}{'2D RMSE':>10}{'Z RMSE':>10}{'3D RMSE':>10}{'Max':>10}")
    for etiket, m in sonuclar:
        print(f"{etiket:<22}{m['rmse_2d']:>9.1f}m{m['rmse_z']:>9.1f}m"
              f"{m['rmse_3d']:>9.1f}m{m['errors_3d'].max():>9.1f}m")


if __name__ == "__main__":
    main()
