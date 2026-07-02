import argparse
import dataclasses
import os
import sys

# Proje kökünü path'e ekle ve çalışma dizini yap (veri yolları köke göredir)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

from vo.sensor_configs import get_config, parse_matlab_calibration
from vo.vo_pipeline import VOPipeline
from vo.reporting import compute_metrics, print_optical_flow_summary, save_csv, plot_trajectory


def main():
    parser = argparse.ArgumentParser(
        description="Monoküler VO — RGB/Termal entegrasyon giriş noktası",
        epilog="Örnek: python3 main.py --sensor thermal --video baska_ucus.mp4 --csv baska_ucus.csv")
    parser.add_argument("--sensor", choices=["rgb", "thermal", "rgb_2025", "thermal_2025"], default="thermal",
                        help="Kullanılacak sensör profili (sensor_configs.py)")
    parser.add_argument("--video", metavar="YOL",
                        help="İşlenecek video dosyası (.mp4/.avi/.mov) veya kare klasörü "
                             "(.webp/.jpg/.png otomatik algılanır). Verilmezse profildeki kullanılır.")
    parser.add_argument("--csv", metavar="YOL",
                        help="Ground truth / translation CSV dosyası. Verilmezse profildeki kullanılır.")
    parser.add_argument("--kesilme", type=int, metavar="KARE",
                        help="GPS kesinti simülasyon karesi (-1: hiç kesilmez, 0: GPS referansı yok). "
                             "Verilmezse profildeki kullanılır.")
    parser.add_argument("--cikti", metavar="PNG",
                        help="Grafik çıktı dosyası adı. Verilmezse profildeki kullanılır.")
    parser.add_argument("--kalibrasyon", metavar="TXT",
                        help="MATLAB cameraParameters metin dosyası; fx/fy/cx/cy ve "
                             "calib boyutları profildekinin yerine bu dosyadan okunur.")
    parser.add_argument("--no-open", action="store_true", help="Grafiği otomatik açma")
    args = parser.parse_args()

    print("Hızlı test başlatılıyor...")
    cfg = get_config(args.sensor)

    # Komut satırı geçersiz kılmaları (dosya düzenlemeden farklı veri koşabilmek için)
    overrides = {}
    if args.video is not None:
        overrides["data_dir"] = args.video
    if args.csv is not None:
        overrides["csv_path"] = args.csv
    if args.kesilme is not None:
        overrides["gps_kesilme_karesi"] = args.kesilme
    if args.cikti is not None:
        overrides["out_fig"] = args.cikti
    if args.kalibrasyon is not None:
        calib = parse_matlab_calibration(args.kalibrasyon)
        overrides.update(calib)
        print(f"Kalibrasyon dosyadan okundu ({args.kalibrasyon}): {calib}")
    if overrides:
        cfg = dataclasses.replace(cfg, **overrides)
        print(f"Geçersiz kılınan ayarlar: {', '.join(overrides)}")

    if not os.path.exists(cfg.data_dir):
        print(f"HATA: Veri bulunamadı: {cfg.data_dir}")
        return
    if not os.path.exists(cfg.csv_path):
        print(f"HATA: CSV bulunamadı: {cfg.csv_path}")
        return

    pipeline = VOPipeline(cfg)
    result = pipeline.run()
    if result is None:
        return

    # Yörünge CSV'si grafik dosyasıyla aynı adı taşır — paralel koşularda
    # sabit tek dosyaya yazıp birbirini ezmesinler
    save_csv(result, os.path.splitext(cfg.out_fig)[0] + "_data.csv")

    metrics = None
    if result["has_gps_reference"]:
        metrics = compute_metrics(result)
    else:
        print_optical_flow_summary(result)

    plot_trajectory(result, metrics, cfg.out_fig, confidence_alarm=cfg.confidence_alarm,
                     open_after=not args.no_open)


if __name__ == "__main__":
    main()
