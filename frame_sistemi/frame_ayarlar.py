# ═══════════════════════════════════════════════════════════════════════════
#  FRAME-BY-FRAME SİSTEMİ — AYARLAR
#
#  NORMALDE BU DOSYAYA DOKUNMANIZA GEREK YOK:
#    1. Kareleri  frame_sistemi/kareler_termal/  (veya kareler_rgb/) içine koyun
#    2. calistir_frame_termal.command  dosyasına çift tıklayın
#    → Uzantı (.jpg/.png/.webp...) otomatik algılanır.
#    → Klasörün içine .csv de koyarsanız otomatik bulunur ve GT olarak kullanılır;
#      CSV yoksa salt optik akış modunda çalışır (metrik hesaplanmaz).
#
#  Kamera kalibrasyonu → vo/sensor_configs.py | VO parametreleri → vo/vo_params.py
# ═══════════════════════════════════════════════════════════════════════════

TERMAL = dict(
    frame_dir="frame_sistemi/kareler_termal",  # karelerin konulacağı klasör
    uzanti=None,                               # None = otomatik algıla
    csv_path="2026_Thermal/THYZ_2026_Ornek_Veri_2_Termal_translation.csv",
    #          ↑ klasör İÇİNDE csv yoksa bu kullanılır; o da yoksa salt optik akış
    gps_kesilme_karesi=450,                    # -1: hiç kesilmez, 0: GPS referansı yok
    out_fig="frame_sistemi/cikti/frame_termal.png",
)

RGB = dict(
    frame_dir="frame_sistemi/kareler_rgb",
    uzanti=None,
    csv_path="2026/THYZ_2026_Ornek_Veri_1_translation.csv",
    gps_kesilme_karesi=450,
    out_fig="frame_sistemi/cikti/frame_rgb.png",
)

# CSV olmadan çalışırken kullanılacak varsayılanlar
DEFAULT_PPM = 0.01          # m/px başlangıç tahmini
DEFAULT_HEADING_DEG = 90.0  # Kuzey
