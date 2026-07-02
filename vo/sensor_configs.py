# ═══════════════════════════════════════════════════════════════════════════
#  SENSÖR PROFİLLERİ — veri yolları + KAMERA KALİBRASYONU
#
#  Yeni bir kalibrasyon (MATLAB cameraParameters çıktısı) eklemek için
#  SADECE ŞU 6 SAYIYI aynen kopyalayın, hiçbir orantılama yapmayın:
#
#      MATLAB çıktısı                      →  buradaki alan
#      FocalLength:    [fx  fy]            →  fx, fy
#      PrincipalPoint: [cx  cy]            →  cx, cy
#      ImageSize:      [yükseklik genişlik] →  calib_height, calib_width  (SIRAYA DİKKAT!)
#
#  Sistem, işlenen videonun/karenin gerçek çözünürlüğünü ilk karede kendisi
#  ölçer ve kalibrasyonu otomatik orantılar (4000×3000 fotoğraf kalibrasyonu
#  ile 3840×2160 video gibi farklı çözünürlükler sorun olmaz).
# ═══════════════════════════════════════════════════════════════════════════
import re
from dataclasses import dataclass


def parse_matlab_calibration(txt_path):
    """MATLAB cameraParameters metin çıktısından kalibrasyonu okur.

    Döner: dict(fx, fy, cx, cy, calib_width, calib_height)
    ImageSize MATLAB'da [yükseklik genişlik] sırasındadır — burada çevrilir.
    """
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    def pair(alan):
        m = re.search(alan + r"\s*:\s*\[([0-9.e+\-]+)\s+([0-9.e+\-]+)\]", text)
        if m is None:
            raise ValueError(f"{txt_path}: '{alan}' bulunamadı")
        return float(m.group(1)), float(m.group(2))

    fx, fy = pair("FocalLength")
    cx, cy = pair("PrincipalPoint")
    h, w = pair("ImageSize")
    return dict(fx=fx, fy=fy, cx=cx, cy=cy,
                calib_width=int(w), calib_height=int(h))


@dataclass
class SensorConfig:
    sensor_type: str
    data_dir: str
    csv_path: str
    out_fig: str
    gps_kesilme_karesi: int
    process_scale: float
    fx: float
    fy: float
    cx: float
    cy: float
    calib_width: int = None    # kalibrasyonun yapıldığı görüntünün GENİŞLİĞİ (px)
    calib_height: int = None   # kalibrasyonun yapıldığı görüntünün YÜKSEKLİĞİ (px)
    default_ppm: float = 0.01
    default_heading_deg: float = 90.0
    confidence_alarm: float = 0.4


CONFIGS = {
    # 2026 termal — kendi yılının verisini işaret eder (2026 uçuşuna özel DR
    # düzeltmeli THERMAL parametre seti; başka yılın verisiyle KOŞMAYIN)
    "THERMAL": SensorConfig(
        sensor_type="THERMAL",
        data_dir="yillar/2026/THYZ_2026_Ornek_Veri_2_Termal.MP4",
        csv_path="yillar/2026/THYZ_2026_Ornek_Veri_2_Termal_translation.csv",
        out_fig="video_sistemi/cikti/trajectory_comparison_2026_thermal.png",
        gps_kesilme_karesi=450,
        process_scale=1.0,  # Termal görüntü 640x512, küçültmeye gerek yok
        fx=731.7965,
        fy=732.0172,
        cx=319.2367,
        cy=251.2424,
        calib_width=640,
        calib_height=512,
    ),
    # 2026 RGB — orijinal 2026 kalibrasyonuna geri döndürüldü
    "RGB": SensorConfig(
        sensor_type="RGB",
        data_dir="yillar/2026/THYZ_2026_Ornek_Veri_1.MP4",
        csv_path="yillar/2026/THYZ_2026_Ornek_Veri_1_translation.csv",
        out_fig="video_sistemi/cikti/trajectory_comparison_2026_v1.png",
        gps_kesilme_karesi=450,
        process_scale=0.5,
        fx=1389.7,
        fy=1387.1,
        cx=954.007,
        cy=558.896,
        calib_width=1920,
        calib_height=1080,
    ),
    # 2025 RGB kamerası — RGB_Kalibrasyon_Parametreleri_2025.txt'den AYNEN kopyalandı
    # (4000×3000 fotoğraf kalibrasyonu; video çözünürlüğüne orantılama otomatik)
    "RGB_2025": SensorConfig(
        sensor_type="RGB",
        data_dir="yillar/2025/Ornek-Veri-2-RGB.MP4",
        csv_path="yillar/2025/Ornek-Veri-2-RGB-translation.csv",
        out_fig="video_sistemi/cikti/trajectory_comparison_2025.png",
        gps_kesilme_karesi=450,
        process_scale=0.5,
        fx=2792.2,          # FocalLength[0]
        fy=2795.2,          # FocalLength[1]
        cx=1988.0,          # PrincipalPoint[0]
        cy=1562.2,          # PrincipalPoint[1]
        calib_width=4000,   # ImageSize: [3000 4000] → genişlik 4000
        calib_height=3000,  #                        → yükseklik 3000
    ),
    # 2025 termal — kalibrasyon 2026 ile aynı kamera (Termal_Kalibrasyon_Parametreleri_2025.txt)
    # ama vo_params'ta NÖTR dead-reckoning profili (THERMAL_2025) kullanır:
    # 2026 uçuşuna özel 0.60 PPM çarpanı / eksen kazançları / dönüş sönümlemesi kapalı
    "THERMAL_2025": SensorConfig(
        sensor_type="THERMAL_2025",
        data_dir="yillar/2025/Ornek-Veri-2-Termal.MP4",
        csv_path="yillar/2025/Ornek-Veri-2-Termal-translation.csv",
        out_fig="video_sistemi/cikti/trajectory_comparison_2025_thermal_fix.png",
        gps_kesilme_karesi=450,
        process_scale=1.0,
        fx=731.7965,
        fy=732.0172,
        cx=319.2367,
        cy=251.2424,
        calib_width=640,
        calib_height=512,
    ),
}


def get_config(sensor_type: str) -> SensorConfig:
    key = sensor_type.upper()
    if key not in CONFIGS:
        raise ValueError(f"Bilinmeyen sensor_type: {sensor_type}. Seçenekler: {list(CONFIGS)}")
    return CONFIGS[key]


###### 2026 Orijinal #####


# # ═══════════════════════════════════════════════════════════════════════════
# #  SENSÖR PROFİLLERİ — veri yolları + KAMERA KALİBRASYONU
# #
# #  Yeni bir kalibrasyon (MATLAB cameraParameters çıktısı) eklemek için
# #  SADECE ŞU 6 SAYIYI aynen kopyalayın, hiçbir orantılama yapmayın:
# #
# #      MATLAB çıktısı                      →  buradaki alan
# #      FocalLength:    [fx  fy]            →  fx, fy
# #      PrincipalPoint: [cx  cy]            →  cx, cy
# #      ImageSize:      [yükseklik genişlik] →  calib_height, calib_width  (SIRAYA DİKKAT!)
# #
# #  Sistem, işlenen videonun/karenin gerçek çözünürlüğünü ilk karede kendisi
# #  ölçer ve kalibrasyonu otomatik orantılar (4000×3000 fotoğraf kalibrasyonu
# #  ile 3840×2160 video gibi farklı çözünürlükler sorun olmaz).
# # ═══════════════════════════════════════════════════════════════════════════
# from dataclasses import dataclass


# @dataclass
# class SensorConfig:
#     sensor_type: str
#     data_dir: str
#     csv_path: str
#     out_fig: str
#     gps_kesilme_karesi: int
#     process_scale: float
#     fx: float
#     fy: float
#     cx: float
#     cy: float
#     calib_width: int = None    # kalibrasyonun yapıldığı görüntünün GENİŞLİĞİ (px)
#     calib_height: int = None   # kalibrasyonun yapıldığı görüntünün YÜKSEKLİĞİ (px)
#     default_ppm: float = 0.01
#     default_heading_deg: float = 90.0
#     confidence_alarm: float = 0.4


# CONFIGS = {
#     "THERMAL": SensorConfig(
#         sensor_type="THERMAL",
#         data_dir="2026_Thermal/THYZ_2026_Ornek_Veri_2_Termal.MP4",
#         csv_path="2026_Thermal/THYZ_2026_Ornek_Veri_2_Termal_translation.csv",
#         out_fig="video_sistemi/cikti/trajectory_comparison_2026_thermal.png",
#         gps_kesilme_karesi=450,
#         process_scale=1.0,  # Termal görüntü 640x512, küçültmeye gerek yok
#         fx=731.7965,
#         fy=732.0172,
#         cx=319.2367,
#         cy=251.2424,
#         calib_width=640,
#         calib_height=512,
#     ),
#     "RGB": SensorConfig(
#         sensor_type="RGB",
#         data_dir="2026/THYZ_2026_Ornek_Veri_1.MP4",
#         csv_path="2026/THYZ_2026_Ornek_Veri_1_translation.csv",
#         out_fig="video_sistemi/cikti/trajectory_comparison_2026_v1.png",
#         gps_kesilme_karesi=450,
#         process_scale=0.5,
#         fx=1389.7,
#         fy=1387.1,
#         cx=954.007,
#         cy=558.896,
#         calib_width=1920,
#         calib_height=1080,
#     ),
#     # 2025 RGB kamerası — RGB_Kalibrasyon_Parametreleri_2025.txt'den AYNEN kopyalandı
#     # (4000×3000 fotoğraf kalibrasyonu; video çözünürlüğüne orantılama otomatik)
#     "RGB_2025": SensorConfig(
#         sensor_type="RGB",
#         data_dir="2025/THYZ_2025_Oturum_2-2",
#         csv_path="2025/THYZ_2025_Oturum_2_Translation.csv",
#         out_fig="video_sistemi/cikti/trajectory_comparison_2025.png",
#         gps_kesilme_karesi=450,
#         process_scale=0.5,
#         fx=2792.2,          # FocalLength[0]
#         fy=2795.2,          # FocalLength[1]
#         cx=1988.0,          # PrincipalPoint[0]
#         cy=1562.2,          # PrincipalPoint[1]
#         calib_width=4000,   # ImageSize: [3000 4000] → genişlik 4000
#         calib_height=3000,  #                        → yükseklik 3000
#     ),
# }


# def get_config(sensor_type: str) -> SensorConfig:
#     key = sensor_type.upper()
#     if key not in CONFIGS:
#         raise ValueError(f"Bilinmeyen sensor_type: {sensor_type}. Seçenekler: {list(CONFIGS)}")
#     return CONFIGS[key]
