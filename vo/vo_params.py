# ═══════════════════════════════════════════════════════════════════════════
#  VO PARAMETRELERİ — ELLE AYAR BÖLGESİ
#
#  Buradaki değerleri değiştirip kaydettikten sonra test etmek için:
#      python3 main.py --sensor thermal --no-open
#      python3 main.py --sensor rgb --no-open
#
#  Mevcut en iyi sonuçlar (2026-07-02):
#      THERMAL: 2D RMSE = 21.71 m   |   RGB: 2D RMSE = 43.99 m
#
#  ⚠ "KİLİTLİ" işaretli parametreler sweet spot'ta — değiştirilince hata
#    belirgin şekilde artıyor. Dokunmadan önce mutlaka tam test koşusu yapın.
# ═══════════════════════════════════════════════════════════════════════════

THERMAL = dict(
    # ── Görüntü ön işleme ───────────────────────────────────────────────────
    clahe_clip_limit=5.0,        # CLAHE kontrast limiti (termal görüntüye uygulanır)

    # ── Özellik tespiti — Shi-Tomasi (KİLİTLİ) ──────────────────────────────
    shi_max_corners=1500,        # köşe sayısı üst sınırı
    shi_quality_level=0.03,      # köşe kalite eşiği
    shi_min_distance=15,         # köşeler arası minimum piksel mesafesi

    # ── Lucas-Kanade optik akış (KİLİTLİ) ───────────────────────────────────
    lk_win_size=41,              # LK pencere boyutu (kare, piksel)

    # ── GPS fazı — öğrenme ──────────────────────────────────────────────────
    heading_alpha=0.50,          # KİLİTLİ — heading EMA öğrenme hızı
    ppm_multiplier_x=0.60,       # GPS fazında öğrenilen PPM'e X çarpanı
    ppm_multiplier_y=0.60,       # GPS fazında öğrenilen PPM'e Y çarpanı

    # ── Dead-reckoning düzeltmeleri (koordinat inişiyle optimize edildi) ────
    # dy_correction = dy_base - dy_slope * dr_progress   (dr_progress: 0→1, 4000 karede)
    dy_base=1.55,                # Y ekseni kazancı, DR başında
    dy_slope=0.40,               # Y kazancının DR boyunca sönümü (1.55 → 1.15)
    # dx_correction = 1.0 + dx_slope * dr_progress
    dx_slope=1.60,               # X ekseni kazancı artışı, DR sonunda 1.0+dx_slope

    # ── Dönüş-farkında XY sönümleme ─────────────────────────────────────────
    # Hızlı yaw dönüşlerinde sahte öteleme akışını bastırır.
    # damp = 1 / (1 + turn_gain * max(turn_ema - turn_thresh, 0))
    turn_thresh=0.20,            # dönüş eşiği (°/kare EMA); altı = düz uçuş, etkisiz
    turn_gain=25.0,              # eşik üstü sönümleme şiddeti

    # ── Z ekseni (irtifa) ───────────────────────────────────────────────────
    alpha_scale=0.1,             # scale→PPM EMA hızı (0.95<scale<1.05 bandında)
    z_mult_down=3.0,             # alçalma düzeltme çarpanı (dinamik, DR ilerledikçe)
    z_mult_up=0.6,               # yükselme düzeltme çarpanı
)

# 2025 termal verisi için NÖTR profil: görüntü işleme (CLAHE, Shi-Tomasi, LK)
# 2026 termal ile aynı (aynı kamera), ama 2026 UÇUŞUNA özel ayarlanan
# dead-reckoning düzeltmeleri (0.60 PPM çarpanı, eksen kazançları, dönüş
# sönümlemesi) kapalı — 2025 testinde bunlar yörüngeyi ~%25'e küçültüyordu.
THERMAL_2025 = dict(
    clahe_clip_limit=5.0,
    shi_max_corners=1500,
    shi_quality_level=0.03,
    shi_min_distance=15,
    lk_win_size=41,
    heading_alpha=0.50,
    ppm_multiplier_x=1.0,
    ppm_multiplier_y=1.0,
    dy_base=1.0,
    dy_slope=0.0,
    dx_slope=0.0,
    turn_thresh=0.20,
    turn_gain=0.0,
    alpha_scale=0.0,             # scale→PPM kapalı (bkz. RGB'deki not)
    z_mult_down=2.0,
    z_mult_up=0.6,
)

RGB = dict(
    # ── Görüntü ön işleme ───────────────────────────────────────────────────
    clahe_clip_limit=None,       # RGB'de CLAHE kullanılmaz

    # ── Özellik tespiti — FAST (RGB'de Shi-Tomasi yerine FAST kullanılır) ──
    fast_threshold=20,           # FAST köşe eşiği
    fast_max_keypoints=3000,     # en güçlü N köşe tutulur

    # ── Lucas-Kanade optik akış ─────────────────────────────────────────────
    lk_win_size=21,

    # ── GPS fazı — öğrenme ──────────────────────────────────────────────────
    heading_alpha=0.50,
    ppm_multiplier_x=1.0,
    ppm_multiplier_y=1.0,

    # ── Dead-reckoning düzeltmeleri ─────────────────────────────────────────
    # RGB'de eksen kazançları ve dönüş sönümlemesi uygulanmaz (nötr değerler).
    dy_base=1.0,
    dy_slope=0.0,
    dx_slope=0.0,
    turn_thresh=0.20,
    turn_gain=0.0,               # 0 = sönümleme kapalı

    # ── Z ekseni (irtifa) ───────────────────────────────────────────────────
    alpha_scale=0.0,             # scale→PPM güncellemesi RGB'de KAPALI: RANSAC scale
                                 # gürültüsü uzun dead-reckoning'de tek yönlü sapıyor ve
                                 # PPM'i eritiyor (2025 testinde 0.0126→0.0069, yörünge
                                 # yarıya küçüldü + Z yukarı sürüklendi). PPM, GPS
                                 # fazında öğrenilen değerde sabit kalır.
    z_mult_down=2.0,
    z_mult_up=0.6,
)

# ── Ortak (sensörden bağımsız) parametreler ────────────────────────────────
COMMON = dict(
    ransac_reproj_threshold=0.6, # KİLİTLİ — affine RANSAC inlier eşiği (piksel)
    lk_max_iters=20,             # KİLİTLİ — LK iterasyon limiti
    lk_epsilon=0.005,            # KİLİTLİ — LK yakınsama eşiği
    yaw_deadband_deg=0.1,        # bu derecenin altındaki kare-başı yaw = gürültü
    yaw_clamp_deg=2.0,           # kare başına fiziksel yaw limiti
    dr_progress_frames=4000.0,   # dr_progress'in 1.0'a ulaştığı DR karesi sayısı
)


def get_params(sensor_type):
    """Sensör tipine göre birleşik parametre sözlüğü döner (COMMON + sensör)."""
    key = sensor_type.upper()
    if key == "THERMAL":
        base = THERMAL
    elif key == "THERMAL_2025":
        base = THERMAL_2025
    else:
        base = RGB
    merged = dict(COMMON)
    merged.update(base)
    return merged



##########2026######## ORİJİNAL

# THERMAL = dict(
#     # ── Görüntü ön işleme ───────────────────────────────────────────────────
#     clahe_clip_limit=5.0,        # CLAHE kontrast limiti (termal görüntüye uygulanır)

#     # ── Özellik tespiti — Shi-Tomasi (KİLİTLİ) ──────────────────────────────
#     shi_max_corners=1500,        # köşe sayısı üst sınırı
#     shi_quality_level=0.03,      # köşe kalite eşiği
#     shi_min_distance=15,         # köşeler arası minimum piksel mesafesi

#     # ── Lucas-Kanade optik akış (KİLİTLİ) ───────────────────────────────────
#     lk_win_size=41,              # LK pencere boyutu (kare, piksel)

#     # ── GPS fazı — öğrenme ──────────────────────────────────────────────────
#     heading_alpha=0.50,          # KİLİTLİ — heading EMA öğrenme hızı
#     ppm_multiplier_x=0.60,       # GPS fazında öğrenilen PPM'e X çarpanı
#     ppm_multiplier_y=0.60,       # GPS fazında öğrenilen PPM'e Y çarpanı

#     # ── Dead-reckoning düzeltmeleri (koordinat inişiyle optimize edildi) ────
#     # dy_correction = dy_base - dy_slope * dr_progress   (dr_progress: 0→1, 4000 karede)
#     dy_base=1.55,                # Y ekseni kazancı, DR başında
#     dy_slope=0.40,               # Y kazancının DR boyunca sönümü (1.55 → 1.15)
#     # dx_correction = 1.0 + dx_slope * dr_progress
#     dx_slope=1.60,               # X ekseni kazancı artışı, DR sonunda 1.0+dx_slope

#     # ── Dönüş-farkında XY sönümleme ─────────────────────────────────────────
#     # Hızlı yaw dönüşlerinde sahte öteleme akışını bastırır.
#     # damp = 1 / (1 + turn_gain * max(turn_ema - turn_thresh, 0))
#     turn_thresh=0.20,            # dönüş eşiği (°/kare EMA); altı = düz uçuş, etkisiz
#     turn_gain=25.0,              # eşik üstü sönümleme şiddeti

#     # ── Z ekseni (irtifa) ───────────────────────────────────────────────────
#     alpha_scale=0.1,             # scale→PPM EMA hızı (0.95<scale<1.05 bandında)
#     z_mult_down=3.0,             # alçalma düzeltme çarpanı (dinamik, DR ilerledikçe)
#     z_mult_up=0.6,               # yükselme düzeltme çarpanı
# )

# RGB = dict(
#     # ── Görüntü ön işleme ───────────────────────────────────────────────────
#     clahe_clip_limit=None,       # RGB'de CLAHE kullanılmaz

#     # ── Özellik tespiti — FAST (RGB'de Shi-Tomasi yerine FAST kullanılır) ──
#     fast_threshold=20,           # FAST köşe eşiği
#     fast_max_keypoints=3000,     # en güçlü N köşe tutulur

#     # ── Lucas-Kanade optik akış ─────────────────────────────────────────────
#     lk_win_size=21,

#     # ── GPS fazı — öğrenme ──────────────────────────────────────────────────
#     heading_alpha=0.50,
#     ppm_multiplier_x=1.0,
#     ppm_multiplier_y=1.0,

#     # ── Dead-reckoning düzeltmeleri ─────────────────────────────────────────
#     # RGB'de eksen kazançları ve dönüş sönümlemesi uygulanmaz (nötr değerler).
#     dy_base=1.0,
#     dy_slope=0.0,
#     dx_slope=0.0,
#     turn_thresh=0.20,
#     turn_gain=0.0,               # 0 = sönümleme kapalı

#     # ── Z ekseni (irtifa) ───────────────────────────────────────────────────
#     alpha_scale=1.3,             # RGB'de scale EMA (bant kontrolü yok)
#     z_mult_down=2.0,
#     z_mult_up=0.6,
# )

# # ── Ortak (sensörden bağımsız) parametreler ────────────────────────────────
# COMMON = dict(
#     ransac_reproj_threshold=0.6, # KİLİTLİ — affine RANSAC inlier eşiği (piksel)
#     lk_max_iters=20,             # KİLİTLİ — LK iterasyon limiti
#     lk_epsilon=0.005,            # KİLİTLİ — LK yakınsama eşiği
#     yaw_deadband_deg=0.1,        # bu derecenin altındaki kare-başı yaw = gürültü
#     yaw_clamp_deg=2.0,           # kare başına fiziksel yaw limiti
#     dr_progress_frames=4000.0,   # dr_progress'in 1.0'a ulaştığı DR karesi sayısı
# )


# def get_params(sensor_type):
#     """Sensör tipine göre birleşik parametre sözlüğü döner (COMMON + sensör)."""
#     base = THERMAL if sensor_type.upper() == "THERMAL" else RGB
#     merged = dict(COMMON)
#     merged.update(base)
#     return merged
