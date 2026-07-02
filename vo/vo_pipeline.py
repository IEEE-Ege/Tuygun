import time

import cv2
import numpy as np

from vo.mono_vo import MonocularVO
from vo.dataset_loader import load_dataset, gps_health_for_row


class VOPipeline:
    """Sensor-agnostic runner: SensorConfig + MonocularVO'yu bağlar, frame döngüsünü yürütür."""

    def __init__(self, config):
        self.config = config
        self.vo = MonocularVO(sensor_type=config.sensor_type)
        self._calib_applied = False
        # calib_width/height verilmemişse eski davranış: değerler doğrudan
        # process_scale ile ölçeklenir (ilk kare beklenmeden).
        if config.calib_width is None or config.calib_height is None:
            fx = config.fx * config.process_scale
            fy = config.fy * config.process_scale
            cx = config.cx * config.process_scale
            cy = config.cy * config.process_scale
            self._set_calibration(fx, fy, cx, cy)
            self._calib_applied = True

    def _set_calibration(self, fx, fy, cx, cy):
        self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        self.vo.K = self.K
        self.vo.update_calibration(focal=fx, pp=(cx, cy))
        print(f"Kalibrasyon — fx={fx:.1f}  fy={fy:.1f}  cx={cx:.1f}  cy={cy:.1f}")

    def apply_frame_calibration(self, gray):
        """İlk işlenen karenin gerçek boyutuna göre kalibrasyonu otomatik orantılar.

        Kalibrasyon hangi çözünürlükte yapılmış olursa olsun (örn. 4000×3000
        fotoğraf), işlenen kare boyutuna (video çözünürlüğü × process_scale)
        eksen bazında ölçeklenir. Her koşuda ilk karede bir kez çağrılır.
        """
        if self._calib_applied:
            return
        cfg = self.config
        h, w = gray.shape[:2]
        sx = w / cfg.calib_width
        sy = h / cfg.calib_height
        print(f"Kalibrasyon orantılama: {cfg.calib_width}x{cfg.calib_height} → "
              f"{w}x{h} (sx={sx:.4f}, sy={sy:.4f})")
        self._set_calibration(cfg.fx * sx, cfg.fy * sy, cfg.cx * sx, cfg.cy * sy)
        self._calib_applied = True

    def run(self):
        cfg = self.config
        df, is_video, video_path = load_dataset(cfg.csv_path, cfg.data_dir)
        has_health_col = "gps_health_status" in df.columns

        gps_all_zero = (
            float(df["translation_x"].max()) == 0.0 and
            float(df["translation_y"].max()) == 0.0
        )
        has_gps_reference = (cfg.gps_kesilme_karesi != 0) and not gps_all_zero

        print(f"Toplam kare: {len(df)} | GPS health kolonu: {'VAR' if has_health_col else 'YOK'}")
        if not has_gps_reference:
            print("  → GPS referansı yok: salt optik akış modu (PPM ve heading varsayılan)")
        elif not has_health_col:
            if cfg.gps_kesilme_karesi < 0:
                print("  → GPS hiç kesilmiyor (sadece GPS fazı testi)")
            else:
                print(f"  → Kare {cfg.gps_kesilme_karesi}'den itibaren GPS kesik simüle edilecek")

        gt_x, gt_y, gt_z = [], [], []
        pred_x, pred_y, pred_z = [], [], []
        conf_list = []
        reliable_list = []
        init_x, init_y, init_z = 0.0, 0.0, 0.0
        gps_cut_frame = None
        first_frame_done = False

        start_time = time.time()
        prev_gps_status = 1

        cap = None
        if is_video:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"HATA: Video açılamadı: {video_path}")
                return None

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
            if cfg.process_scale != 1.0:
                gray = cv2.resize(gray, None, fx=cfg.process_scale, fy=cfg.process_scale,
                                   interpolation=cv2.INTER_AREA)

            self.apply_frame_calibration(gray)

            true_x = float(row["translation_x"])
            true_y = float(row["translation_y"])
            true_z = float(row["translation_z"])

            if frame_idx == 0:
                init_x, init_y, init_z = true_x, true_y, true_z

            health = gps_health_for_row(row, frame_idx, has_health_col, cfg.gps_kesilme_karesi)

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

            if prev_gps_status == 1 and health == 0:
                if cfg.gps_kesilme_karesi != 0:
                    print(f"\n*** GPS KESİLDİ — kare {frame_idx} ***\n")
            prev_gps_status = health

            sonuc = self.vo.process_frame(gray, json_data)

            if not first_frame_done:
                first_frame_done = True
                if not self.vo.ppm_initialized or not self.vo.heading_initialized:
                    self.vo.force_initialize(ppm=cfg.default_ppm, heading_deg=cfg.default_heading_deg)
                    print(f"  → VO varsayılan başlatıldı: PPM={cfg.default_ppm}, Heading={cfg.default_heading_deg}°")

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
                    f"conf={conf:.2f} | PPM_X={self.vo.ppm_x:.5f} PPM_Y={self.vo.ppm_y:.5f} | {h_ok} | {fps:.1f} FPS"
                )

        elapsed = time.time() - start_time
        if cap is not None:
            cap.release()

        return {
            "gt_x": gt_x, "gt_y": gt_y, "gt_z": gt_z,
            "pred_x": pred_x, "pred_y": pred_y, "pred_z": pred_z,
            "conf_list": conf_list, "reliable_list": reliable_list,
            "gps_cut_frame": gps_cut_frame,
            "has_gps_reference": has_gps_reference,
            "elapsed": elapsed,
            "vo": self.vo,
        }
