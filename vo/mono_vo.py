import cv2
import numpy as np
import math

try:
    from vo.vo_params import get_params   # kökten çalıştırılınca (main.py)
except ImportError:
    from vo_params import get_params      # vo/ içinden doğrudan çalıştırılınca


class MonocularVO:
    def __init__(self, min_num_feat=2000, sensor_type="RGB"):
        self.min_num_feat = min_num_feat
        self.sensor_type = sensor_type
        # Elle ayarlanabilir tüm parametreler vo_params.py'de toplanmıştır
        self.P = get_params(sensor_type)

        self.K = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        self.focal = 1388.4
        self.pp = (954.0, 558.9)
        self.z_offset = 0.0
        self.dr_start_z = None

        self.px_ref = None
        self.px_cur = None
        self.prev_frame = None
        self.cur_frame = None

        self.R_f = np.eye(3, dtype=np.float64)
        self.t_f = np.zeros((3, 1), dtype=np.float64)

        # GPS başlangıç konumu
        self.initial_gps = None

        # ── PPM (piksel→metre) — EMA ─────────────────────────────────────────
        self.ppm_x = 0.10
        self.ppm_y = 0.10
        self.ppm_initialized = False
        self.ppm_alpha = 0.15
        # GPS fazında PPM/heading öğrenimine kanıt sağlayan toplam GPS yolu (m).
        # Bu eşiğin altında kalırsa (örn. GPS fazında hover) ölçek ÖĞRENİLMEMİŞ
        # demektir; dead-reckoning varsayılan PPM ile yapılır ve confidence düşer.
        self.gps_motion_total = 0.0
        self.SCALE_LEARN_MIN_DIST = 5.0
        # Yavaş/yüksek irtifalı uçuşlarda kare başına akış öğrenme eşiğinin
        # altında kalır (2024: ~1.5 px/kare < 3 px eşik → hiç öğrenme olmuyordu).
        # Piksel deltaları eşik aşılana kadar kareler boyunca BİRİKTİRİLİR.
        self._acc_dx = 0.0
        self._acc_dy = 0.0
        self._acc_frames = 0
        self._acc_start_gps = None

        # ── Heading (yön) takibi ─────────────────────────────────────────────
        #
        # heading_angle: drone'un mevcut yönü, standart matematik açısı
        #   0    = Doğu,  π/2 = Kuzey,  π = Batı,  -π/2 = Güney
        #
        # GPS fazında GPS hız vektöründen öğrenilir:
        #   theta_world = atan2(dGPS_y, dGPS_x)          — dünya yönü
        #   theta_pixel = atan2(-dy, dx)                  — piksel hareket yönü
        #   heading_angle = theta_world - theta_pixel + π/2
        #
        # Dead-reckoning'de heading dondurulur (yaw güncellenmez).
        # Optical flow yaw'u nadir kameralar için düşük SNR'lıdır;
        # birikimli drift, sabit yönden daha büyük hata verir.
        #
        # Dönüşüm (piksel → dünya):
        #   phi = heading_angle - π/2
        #   pixel_unit = [dx/|d|,  -dy/|d|]   ← görüntü y ekseni ters
        #   world_unit = R(phi) @ pixel_unit
        #   world_delta = PPM * |d| * world_unit
        # ── Chirality (yansıma) tespiti ──────────────────────────────────────
        # Bazı yıllarda (2024 verisi) kamera/CSV eksen konvansiyonu aynalı:
        # piksel→dünya dönüşümü rotasyon değil YANSIMA gerektirir. GPS fazında
        # iki hipotez paralel izlenir; heading'i iç tutarlı olan seçilir.
        #   chirality = +1 → normal (dy işareti ters, mevcut konvansiyon)
        #   chirality = -1 → aynalı (dy işareti düz, yaw yönü de ters)
        self.chirality = 1.0
        self._chir_h = {1: None, -1: None}     # hipotez başına heading EMA
        self._chir_err = {1: None, -1: None}   # hipotez başına |yenilik| EMA
        self._chir_n = 0

        self.heading_angle = None
        self.heading_initialized = False
        self.heading_alpha = self.P["heading_alpha"]  # EMA ağırlığı (GPS güncellemesi)
        # Birikimli örnek eşikleri: güncelleme ancak İKİSİ de sağlanınca yapılır.
        # GPS eşiği 0.5 m — hover jitter'ı (mm/kare) PPM'i zehirlemesin (SNR).
        # 0.2 m: hover jitter'ı eler (100 karelik pencerede ~0.17 m < eşik)
        # ama normal uçuşta sık güncellemeye izin verir. 0.5 çok seyrekleştirip
        # 2025 RGB V1 kazancını kaybettiriyordu, 0.05 hover çöpü topluyordu.
        self.heading_gps_min_dist = 0.2   # birikmiş GPS deltası (m)
        self.heading_pix_min_disp = 3.0   # birikmiş piksel deltası (px)
        self.ACC_MAX_FRAMES = 100  # pencere yaş sınırı: eşiklere bu sürede
                                   # ulaşılamazsa (hover) segment atılır; uzun
                                   # pencerede dönme birikimi düz toplamayı bozar

        # ── Güvenilirlik takibi ──────────────────────────────────────────────
        self.dead_reckoning_active = False
        self.dr_frame_count = 0
        self.last_inlier_ratio = 1.0
        self.last_feature_count = 0
        self.last_affine_scale = 1.0
        self.last_pixel_velocity = 0.0
        # Birikimli yaw: GPS kesilmesinden bu yana toplam rotasyon (radyan)
        # Büyük birikim → heading güvensizliği artar
        self.accumulated_dr_yaw = 0.0

        # Eşikler
        self.THRESH_INLIER_RATIO  = 0.25
        self.THRESH_MIN_FEATURES  = 50
        self.THRESH_MAX_SCALE_DEV = 0.6
        self.THRESH_MAX_VEL_MF    = 6.0
        self.THRESH_MAX_DR_FRAMES = 2000
        # Birikimli yaw eşiği: bu değeri geçince confidence düşmeye başlar
        self.THRESH_ACCUM_YAW     = math.pi  # 180° birikim

        # Legacy
        self.last_valid_scale = 1.0
        self.last_valid_velocity = 0.0
        self.frames_since_keyframe = 0

        self.prev_json_data = None
        self.keyframe_json = None
        self.is_initialized = False

        # Termal görüntüler için kontrast artırıcı CLAHE objesi
        clip = self.P["clahe_clip_limit"]
        self.clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)) if clip is not None else None

        # ── Detektör ve Parametreler ─────────────────────────────────────────────
        self.detector = cv2.FastFeatureDetector_create(
            threshold=self.P.get("fast_threshold", 10), nonmaxSuppression=True)

        win = self.P["lk_win_size"]
        self.lk_params = dict(winSize=(win, win),
                              maxLevel=5,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                                        self.P["lk_max_iters"], self.P["lk_epsilon"]))

    def update_calibration(self, focal, pp):
        self.focal = focal
        self.pp = pp

    def force_initialize(self, ppm=0.01, heading_deg=90.0):
        """GPS verisi olmayan senaryolarda PPM ve heading'i varsayılanla başlatır."""
        if ppm > 0:
            self.ppm_x = ppm
            self.ppm_y = ppm
            self.ppm_initialized = True
        if not self.heading_initialized:
            self.heading_angle = math.radians(heading_deg)
            self.heading_initialized = True

    # ── Özellik tespiti & takibi ─────────────────────────────────────────────

    def feature_detection(self, img):
        h, w = img.shape
        margin_y = int(h * 0.15)
        margin_x = int(w * 0.15)
        mask = np.zeros_like(img)
        mask[margin_y:h-margin_y, margin_x:w-margin_x] = 255
        
        if self.sensor_type.startswith("THERMAL"):
            # Termal kameralarda ortadaki sabit crosshair'i (hedef imlecini) maskele
            cy, cx = h // 2, w // 2
            mask[cy-40:cy+40, cx-40:cx+40] = 0
            
            # Termal için gürültüye daha dayanıklı Shi-Tomasi kullan (daha az ama çok güçlü noktalar)
            corners = cv2.goodFeaturesToTrack(img,
                                              maxCorners=self.P["shi_max_corners"],
                                              qualityLevel=self.P["shi_quality_level"],
                                              minDistance=self.P["shi_min_distance"],
                                              mask=mask)
            if corners is not None:
                return corners.reshape(-1, 2)
            return np.empty((0, 2), dtype=np.float32)

        keypoints = self.detector.detect(img, mask=mask)
        if not keypoints:
            return np.empty((0, 2), dtype=np.float32)
        keypoints = sorted(keypoints, key=lambda kp: kp.response,
                           reverse=True)[:self.P.get("fast_max_keypoints", 3000)]
        return np.array([kp.pt for kp in keypoints], dtype=np.float32)

    def feature_tracking(self, img_ref, img_cur, px_ref):
        kp2, st, err = cv2.calcOpticalFlowPyrLK(img_ref, img_cur, px_ref, None, **self.lk_params)
        st = st.reshape(st.shape[0])
        return px_ref[st == 1], kp2[st == 1]

    # ── Yardımcılar ──────────────────────────────────────────────────────────

    @staticmethod
    def _wrap_angle(a):
        """Açıyı [-π, π] aralığına taşır."""
        return math.atan2(math.sin(a), math.cos(a))

    def _circular_ema(self, current, new_val, alpha):
        """Dairesel EMA: açıları doğru interpolasyonla karıştırır."""
        delta = self._wrap_angle(new_val - current)
        return self._wrap_angle(current + alpha * delta)

    # ── GPS fazı: PPM ve heading güncelleme ──────────────────────────────────

    def _update_ppm_and_heading(self, dx, dy, pixel_displacement, json_data):
        """
        GPS sağlıklıyken her keyframe geçişinde çağrılır.
        PPM ve heading_angle'ı GPS + optik akış bilgisiyle günceller.
        """
        if self.keyframe_json is None:
            return

        # ── Piksel delta birikimi ────────────────────────────────────────────
        # Kare başına akış eşik altında kalabilir (yavaş uçuş); eşik aşılana
        # kadar biriktir, sinyal yeterince toplanınca TEK güncelleme yap.
        # (Kare-arası rotasyon <2° kelepçeli olduğundan düz toplama geçerlidir.)
        if self._acc_start_gps is None:
            self._acc_start_gps = (self.keyframe_json.get("translation_x", 0.0),
                                   self.keyframe_json.get("translation_y", 0.0))
            self._acc_frames = 0
        self._acc_dx += dx
        self._acc_dy += dy
        self._acc_frames += 1
        acc_disp = math.hypot(self._acc_dx, self._acc_dy)

        gps_dx = json_data.get("translation_x", 0.0) - self._acc_start_gps[0]
        gps_dy = json_data.get("translation_y", 0.0) - self._acc_start_gps[1]
        gps_dist = math.sqrt(gps_dx**2 + gps_dy**2)

        if acc_disp < self.heading_pix_min_disp or gps_dist < self.heading_gps_min_dist:
            # Henüz yeterli sinyal yok — biriktirmeye devam; ama pencere çok
            # yaşlandıysa (hover / dönüş birikimi) segmenti at
            if self._acc_frames >= self.ACC_MAX_FRAMES:
                self._acc_dx = self._acc_dy = 0.0
                self._acc_start_gps = None
            return

        # Bu güncellemede birikmiş deltalar kullanılır, sonra birikim sıfırlanır
        dx = self._acc_dx
        dy = self._acc_dy
        pixel_displacement = acc_disp
        self._acc_dx = self._acc_dy = 0.0
        self._acc_frames = 0
        self._acc_start_gps = (json_data.get("translation_x", 0.0),
                               json_data.get("translation_y", 0.0))

        self.gps_motion_total += gps_dist

        ppm_general = gps_dist / pixel_displacement

        new_ppm_y = ppm_general * self.P["ppm_multiplier_y"]
        new_ppm_x = ppm_general * self.P["ppm_multiplier_x"]

        # PPM güncelle
        if not self.ppm_initialized:
            self.ppm_x = new_ppm_x
            self.ppm_y = new_ppm_y
            self.ppm_initialized = True
        else:
            self.ppm_x = ((1.0 - self.ppm_alpha) * self.ppm_x + self.ppm_alpha * new_ppm_x)
            self.ppm_y = ((1.0 - self.ppm_alpha) * self.ppm_y + self.ppm_alpha * new_ppm_y)

        # Heading güncelle
        # theta_world : GPS hareket yönü (standart açı)
        # theta_pixel : metrik olarak ölçeklenmiş piksel hareket yönü
        # heading_angle = theta_world - theta_pixel + π/2
        theta_world = math.atan2(gps_dy, gps_dx)

        # ── Chirality hipotez takibi ─────────────────────────────────────────
        # Doğru hipotezde heading tahmini gidiş yönünden bağımsız tutarlıdır;
        # yanlış hipotezde her dönüşte zıplar (yenilik hatası büyür).
        for s in (1, -1):
            tp = math.atan2(-s * dy * new_ppm_y, dx * new_ppm_x)
            h = self._wrap_angle(theta_world - tp + math.pi / 2)
            onceki = self._chir_h[s]
            if onceki is None:
                self._chir_h[s] = h
            else:
                yenilik = abs(self._wrap_angle(h - onceki))
                e = self._chir_err[s]
                self._chir_err[s] = yenilik if e is None else 0.9 * e + 0.1 * yenilik
                self._chir_h[s] = self._circular_ema(onceki, h, 0.5)
        self._chir_n += 1
        if (self._chir_n >= 20
                and self._chir_err[1] is not None and self._chir_err[-1] is not None):
            aktif = int(self.chirality)
            diger = -aktif
            # Histerezis: ancak diğer hipotez belirgin (2x) tutarlıysa geç
            if self._chir_err[diger] < 0.5 * self._chir_err[aktif]:
                print(f"  [VO] YANSIMA TESPİTİ: chirality {aktif:+d} → {diger:+d} "
                      f"(yenilik hatası {self._chir_err[aktif]:.3f} → {self._chir_err[diger]:.3f} rad)")
                self.chirality = float(diger)
                self.heading_angle = self._chir_h[diger]

        theta_pixel = math.atan2(-self.chirality * dy * new_ppm_y, dx * new_ppm_x)
        new_heading = self._wrap_angle(theta_world - theta_pixel + math.pi / 2)

        if not self.heading_initialized:
            self.heading_angle = new_heading
            self.heading_initialized = True
        else:
            self.heading_angle = self._circular_ema(
                self.heading_angle, new_heading, self.heading_alpha)

    # ── Dead-reckoning: piksel → dünya ──────────────────────────────────────

    def _dead_reckon(self, m, dx, dy, pixel_displacement):
        """
        Optik akış afin matrisi kullanarak pozisyonu günceller.
        1. heading_angle'ı yaw bileşeniyle günceller.
        2. Piksel deplasmanını dünya koordinatlarına döndürür.
        """
        if not self.heading_initialized or not self.ppm_initialized:
            return  # yeterli bilgi yok

        # Yaw: afin matrisin rotasyon bileşeni
        # M = cur->ref dönüşümü, drone CW döndüğünde M CCW açısı verir.
        yaw = math.atan2(m[1, 0], m[0, 0])
        # Soft deadband: eşik altındaki dönüşleri gürültü kabul et
        deadband = math.radians(self.P["yaw_deadband_deg"])
        if abs(yaw) < deadband:
            yaw = 0.0

        # Fiziksel yaw limiti (°/kare)
        clamp = math.radians(self.P["yaw_clamp_deg"])
        yaw = max(-clamp, min(clamp, yaw))
        # Aynalı geometride görüntüden ölçülen yaw'un dünya yönü de terstir
        self.heading_angle = self._wrap_angle(self.heading_angle - self.chirality * yaw)
        self.accumulated_dr_yaw += abs(yaw)  # heading güvensizliği takibi

        # Güncel AGL irtifamızı hesaplayalım (düzeltilmiş Z ekseninden)
        # GPS hiç görülmediyse (salt optik akış modu) başlangıç Z'si 0 kabul edilir
        init_z = self.initial_gps[2] if self.initial_gps is not None else 0.0
        current_z_agl = max(self.z_offset - self.t_f[2][0] - init_z, 1.0)
        # Optik akıştan gelen ham AGL irtifası
        raw_z_agl = max(self.focal * ((self.ppm_x + self.ppm_y) / 2.0), 1.0)

        # Düzeltilmiş irtifaya göre PPM çarpanı (irtifa arttıkça PPM de artmalı)
        # Kelepçe: Z ekseni sürüklenirse current_z_agl 1 m tabanına çakılıp
        # çarpanı ~0 yapıyor ve XY tahmini tamamen donuyordu (2025 testi).
        # İrtifa kompanzasyonu en fazla ±2x düzeltme yapabilir.
        z_multiplier = min(max(current_z_agl / raw_z_agl, 0.5), 2.0)

        # Kamera frame'indeki metrik hareket (irtifa düzeltmeli):
        dx_meter = dx * (self.ppm_x * z_multiplier)
        dy_meter = -self.chirality * dy * (self.ppm_y * z_multiplier)

        if self.sensor_type.startswith("THERMAL"):
            # Eksen kazançları: DR ilerledikçe (dr_progress 0→1) doğrusal değişir
            # (THERMAL_2025 profilinde katsayılar nötr — blok etkisiz)
            dr_progress = min(self.dr_frame_count / self.P["dr_progress_frames"], 1.0)
            dy_correction = self.P["dy_base"] - self.P["dy_slope"] * dr_progress
            dy_meter *= dy_correction
            dx_correction = 1.0 + self.P["dx_slope"] * dr_progress
            dx_meter *= dx_correction

            # ── Dönüş-farkında XY sönümleme ──────────────────────────────────
            # Hızlı yaw dönüşlerinde (kamera tam nadir olmadığından) dönüş,
            # öteleme benzeri sahte akış üretir; VO gerçekte olmayan XY hareketi
            # ölçer (analiz: dönüş pencerelerinde 1.5-4.6x aşırı ölçüm, düz
            # uçuşta oran ~1.0). Ham |yaw|'ın EMA'sı dönüş dedektörüdür; eşik
            # normal uçuş seviyesinin üstünde olduğundan düz uçuş hiç etkilenmez,
            # sadece belirgin dönüşlerde sönümleme devreye girer.
            raw_yaw_deg = abs(math.degrees(math.atan2(m[1, 0], m[0, 0])))
            self.turn_ema = 0.95 * getattr(self, "turn_ema", 0.0) + 0.05 * raw_yaw_deg
            turn_excess = max(self.turn_ema - self.P["turn_thresh"], 0.0)
            maneuver_damp = 1.0 / (1.0 + self.P["turn_gain"] * turn_excess)
            dx_meter *= maneuver_damp
            dy_meter *= maneuver_damp

        # Dünya birim vektörü hesabını kamera metre hesabıyla yapalım:
        phi = self.heading_angle - math.pi / 2
        c, s = math.cos(phi), math.sin(phi)
        
        world_dx = c * dx_meter - s * dy_meter
        world_dy = s * dx_meter + c * dy_meter

        # Fizik dışı atlamayı sınırla (kalıbozuk optik akış)
        # DR ilerledikçe giderek daha sıkı sınırlandır (drift birikmesini yavaşlat)
        dr_velocity_cap = max(self.THRESH_MAX_VEL_MF - (self.dr_frame_count / 3000.0) * 2.0, 2.5)
        dist = math.sqrt(world_dx**2 + world_dy**2)
        if dist > dr_velocity_cap:
            ratio = dr_velocity_cap / dist
            world_dx *= ratio
            world_dy *= ratio

        # t_f güncelle:  x_val = t_f[0],  y_val = -t_f[1]
        self.t_f[0][0] += world_dx
        self.t_f[1][0] -= world_dy   # y_val = -t_f[1] → t_f[1] = -y_val

        # Scale tabanlı Z-ekseni (İrtifa) tahmini
        # Scale = Z_cur / Z_prev. scale < 1 ise dron yükseliyor (nesneler küçülür), scale > 1 ise alçalıyor.
        # m = cv2.estimateAffinePartial2D(px_cur, px_ref) -> scale < 1 demek px_cur > px_ref (büyüme var, alçalma)
        scale = math.sqrt(m[0, 0]**2 + m[1, 0]**2)

        if 0.90 < scale < 1.30 and scale != 1.0:
            # Gürültü ve ani sıçramaları önlemek için Exponential Moving Average (EMA)
            # Bant kontrolü artık her sensörde: banttan uzak scale değerleri gürültü/
            # manevra kaynaklıdır, PPM'e işlenirse binlerce karede tek yönlü aşınma
            # yapıp Z'yi sürüklüyordu (2025 testindeki donmanın kök nedeni).
            if 0.95 < scale < 1.05:
                alpha_scale = self.P["alpha_scale"]
            else:
                alpha_scale = 0.0
            smoothed_scale = 1.0 + (scale - 1.0) * alpha_scale
            # PPM değişimini kare başına ±2% ile sınırla (drift kartopu etkisini daha iyi engelle)
            smoothed_scale = max(0.98, min(1.02, smoothed_scale))
            self.ppm_x *= smoothed_scale
            self.ppm_y *= smoothed_scale

        # Z ekseni için saf ppm kullanılarak raw hesaplama
        current_z_agl = self.focal * ((self.ppm_x + self.ppm_y) / 2.0)
        # Z Ekseni, dataset'te muhtemelen Aşağı-Pozitif (NED) veya zıt yönlü. AGL ise Yukarı-Pozitif.
        # Bu yüzden AGL'yi eksi (-) ile işleme alıyoruz ki mirror (ayna) efekti düzelsin.
        raw_vo_z = self.z_offset - current_z_agl - init_z
        
        # Dead Reckoning Sırasında Z Ekseni Asimetrik Düzeltmesi (Geometrik Drift Yaratmaz!)
        # Dron yükselirken (raw_vo_z azalır) optik akış asimetrik hata yapar ve over-estimate eder -> x0.6
        # Dron alçalırken (raw_vo_z artar) optik akış under-estimate eder -> x3.0
        if self.dr_start_z is not None:
            delta_z = raw_vo_z - self.dr_start_z
            
            # Kullanıcının uyarısı: "oran orantı katsayısı biraz fazla geniş"
            # 3.0 çarpanı 450-2000 arasındaki yalancı çukuru (false dip) çok büyütüyordu (V şekli yapıyordu).
            # Çarpanı 1.2'ye düşürerek hem doğru yönde kalmasını hem de yalancı çukurun düzleşmesini sağlıyoruz.
            if delta_z < 0:
                # Zamanla biriken "scale drift" (survival bias) nedeniyle Z ekseni alçalmaları giderek
                # daha fazla under-estimate edilir (az hesaplanır). 
                # Bunu telafi etmek için dr_frame_count'a bağlı dinamik bir çarpan (1.0 -> 3.0) kullanıyoruz.
                drift_compensation = min(self.dr_frame_count / 3000.0, 1.0)
                dynamic_multiplier = 1.0 + drift_compensation * self.P["z_mult_down"]
                corrected_delta_z = delta_z * dynamic_multiplier
            else:
                corrected_delta_z = delta_z * self.P["z_mult_up"]  # Yükselmeyi / tepeyi küçült
                
            self.t_f[2][0] = self.dr_start_z + corrected_delta_z
        else:
            self.t_f[2][0] = raw_vo_z

    # ── GPS pozisyon güncellemesi ────────────────────────────────────────────

    def get_absolute_scale(self, cur_json, keyframe_json, pixel_displacement=0.0):
        """Geriye dönük uyumluluk için tutuldu; PPM artık _update_ppm_and_heading içinde."""
        if cur_json.get("gps_health_status", 0) == 1 and keyframe_json is not None:
            x_prev = keyframe_json.get("translation_x", 0.0)
            y_prev = keyframe_json.get("translation_y", 0.0)
            z_prev = keyframe_json.get("translation_z", 0.0)
            x_cur  = cur_json.get("translation_x", 0.0)
            y_cur  = cur_json.get("translation_y", 0.0)
            z_cur  = cur_json.get("translation_z", 0.0)
            scale = math.sqrt((x_cur-x_prev)**2 + (y_cur-y_prev)**2 + (z_cur-z_prev)**2)
            self.last_valid_scale = scale
            return scale
        return max(pixel_displacement * ((self.ppm_x + self.ppm_y) / 2.0), 0.0)

    def _update_from_gps(self, json_data):
        if self.initial_gps is None:
            return
        self.t_f[0][0] = json_data.get("translation_x", 0.0) - self.initial_gps[0]
        self.t_f[1][0] = -(json_data.get("translation_y", 0.0) - self.initial_gps[1])
        self.t_f[2][0] = json_data.get("translation_z", 0.0) - self.initial_gps[2]
        
        # GPS verisi geldiği sürece dr_start_z güncellenir
        # GPS kesildiğinde en son bu değerde donup kalır
        self.dr_start_z = self.t_f[2][0]
        
        # Z ofsetini güncelle (GPS varken PPM hesaplanıyor, GPS Z ve AGL ters yönlü)
        if self.ppm_initialized:
            current_z_agl = self.focal * ((self.ppm_x + self.ppm_y) / 2.0)
            self.z_offset = self.t_f[2][0] + self.initial_gps[2] + current_z_agl

    # ── Güvenilirlik skoru ───────────────────────────────────────────────────

    def _compute_confidence(self):
        if not self.dead_reckoning_active:
            return 1.0

        score = 1.0

        # Ölçek hiç öğrenilemediyse (GPS fazında yeterli hareket yoktu) konum
        # tahmini varsayılan PPM/heading'e dayanır — güven ciddi şekilde düşmeli
        if self.gps_motion_total < self.SCALE_LEARN_MIN_DIST:
            score *= 0.3

        if self.last_inlier_ratio < self.THRESH_INLIER_RATIO:
            score *= (self.last_inlier_ratio / self.THRESH_INLIER_RATIO) * 0.5
        else:
            bonus = min((self.last_inlier_ratio - self.THRESH_INLIER_RATIO)
                        / (1.0 - self.THRESH_INLIER_RATIO), 1.0)
            score *= 0.8 + 0.2 * bonus

        if self.last_feature_count < self.THRESH_MIN_FEATURES:
            score *= max(self.last_feature_count / self.THRESH_MIN_FEATURES, 0.0)

        scale_dev = abs(self.last_affine_scale - 1.0)
        if scale_dev > self.THRESH_MAX_SCALE_DEV:
            penalty = (scale_dev - self.THRESH_MAX_SCALE_DEV) / self.THRESH_MAX_SCALE_DEV
            score *= max(1.0 - penalty, 0.0)

        if self.last_pixel_velocity * ((self.ppm_x + self.ppm_y) / 2.0) > self.THRESH_MAX_VEL_MF:
            score *= 0.3

        if self.dr_frame_count > self.THRESH_MAX_DR_FRAMES:
            decay = max(1.0 - (self.dr_frame_count - self.THRESH_MAX_DR_FRAMES) / 2000.0, 0.6)
            score *= decay

        # Birikimli yaw penaltısı: heading gürültüsünün birikmesi konum belirsizliği yaratır
        if self.accumulated_dr_yaw > self.THRESH_ACCUM_YAW:
            excess = self.accumulated_dr_yaw - self.THRESH_ACCUM_YAW
            # Her ek 180°'lik birikim için 15% düşüş, minimum 0.6
            yaw_decay = max(1.0 - (excess / math.pi) * 0.15, 0.6)
            score *= yaw_decay

        return float(np.clip(score, 0.0, 1.0))

    # ── Ana işlem döngüsü ───────────────────────────────────────────────────

    def process_frame(self, image, json_data):
        """
        image    : Gri tonlamalı OpenCV görüntüsü.
        json_data: {"translation_x": x, "translation_y": y, "translation_z": z,
                    "gps_health_status": 0|1}

        Dönen dict anahtarları:
          detected_translations[0].translation_x/y/z
          confidence, is_reliable, dead_reckoning, dr_frame_count,
          inlier_ratio, feature_count, heading_deg (debug)
        """
        if self.clahe is not None:
            self.cur_frame = self.clahe.apply(image)
        else:
            self.cur_frame = image
            
        self.frames_since_keyframe += 1

        gps_healthy = json_data.get("gps_health_status", 0) == 1

        if gps_healthy:
            self.dead_reckoning_active = False
            self.dr_frame_count = 0
            self.accumulated_dr_yaw = 0.0  # GPS geri gelince sıfırla
        else:
            self.dead_reckoning_active = True
            self.dr_frame_count += 1
            # GPS yokken piksel birikimi anlamsız — sıfırla
            self._acc_dx = self._acc_dy = 0.0
            self._acc_start_gps = None

        if gps_healthy and self.initial_gps is None:
            self.initial_gps = (
                json_data.get("translation_x", 0.0),
                json_data.get("translation_y", 0.0),
                json_data.get("translation_z", 0.0),
            )

        # ── İlk kare ────────────────────────────────────────────────────────
        if not self.is_initialized:
            self.px_ref = self.feature_detection(self.cur_frame)
            self.prev_frame = self.cur_frame.copy()
            self.prev_json_data = json_data
            self.keyframe_json = json_data
            self.is_initialized = True
            if gps_healthy and self.initial_gps is not None:
                self._update_from_gps(json_data)
            return self._format_output()

        # ── Özellik kontrolü ────────────────────────────────────────────────
        if len(self.px_ref) == 0:
            self.px_ref = self.feature_detection(self.prev_frame)
            if len(self.px_ref) == 0:
                self.prev_frame = self.cur_frame.copy()
                self.prev_json_data = json_data
                self.keyframe_json = json_data
                if gps_healthy and self.initial_gps is not None:
                    self._update_from_gps(json_data)
                return self._format_output()

        self.px_ref, self.px_cur = self.feature_tracking(
            self.prev_frame, self.cur_frame, self.px_ref)
        self.last_feature_count = len(self.px_ref)

        if len(self.px_ref) < 8:
            self.px_cur = self.feature_detection(self.cur_frame)
            self.px_ref = self.px_cur
            self.prev_frame = self.cur_frame.copy()
            self.prev_json_data = json_data
            self.keyframe_json = json_data
            if gps_healthy and self.initial_gps is not None:
                self._update_from_gps(json_data)
            return self._format_output()

        # ── Afin tahmin ──────────────────────────────────────────────────────
        px_cur_c = self.px_cur - np.array(self.pp)
        px_ref_c = self.px_ref - np.array(self.pp)

        m, inliers = cv2.estimateAffinePartial2D(
            px_cur_c, px_ref_c,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.P["ransac_reproj_threshold"],
            maxIters=2000,
            confidence=0.99
        )

        if m is None:
            if gps_healthy and self.initial_gps is not None:
                self._update_from_gps(json_data)
            self.prev_json_data = json_data
            return self._format_output()

        inlier_count = int(np.sum(inliers)) if inliers is not None else 0
        self.last_inlier_ratio = inlier_count / max(len(self.px_cur), 1)
        self.last_affine_scale = math.sqrt(m[0, 0]**2 + m[1, 0]**2)

        dx = m[0, 2]
        dy = m[1, 2]
        pixel_displacement = math.sqrt(dx**2 + dy**2)
        self.last_pixel_velocity = pixel_displacement

        if pixel_displacement < 0.5:
            if gps_healthy and self.initial_gps is not None:
                self._update_from_gps(json_data)
            self.prev_json_data = json_data
            return self._format_output()

        # ── Pozisyon güncelleme ──────────────────────────────────────────────
        if gps_healthy and self.initial_gps is not None:
            # GPS SAĞLIKLI: sıfır hata, direkt GPS konumu
            self._update_from_gps(json_data)
            # PPM ve heading öğren
            self._update_ppm_and_heading(dx, dy, pixel_displacement, json_data)

        else:
            # GPS KESİLDİ: heading + PPM ile dead-reckoning
            self._dead_reckon(m, dx, dy, pixel_displacement)

        # ── Keyframe güncelle ────────────────────────────────────────────────
        if self.px_ref.shape[0] < self.min_num_feat:
            self.px_cur = self.feature_detection(self.cur_frame)

        self.prev_frame = self.cur_frame.copy()
        self.px_ref = self.px_cur
        self.prev_json_data = json_data
        self.keyframe_json = json_data
        self.frames_since_keyframe = 0

        return self._format_output()

    def _format_output(self):
        confidence = self._compute_confidence()
        heading_deg = math.degrees(self.heading_angle) if self.heading_initialized else None
        return {
            "detected_translations": [
                {
                    "translation_x": float(self.t_f[0][0]),
                    "translation_y": float(-self.t_f[1][0]),
                    "translation_z": float(self.t_f[2][0])
                }
            ],
            "confidence": confidence,
            "is_reliable": (
                self.last_inlier_ratio >= self.THRESH_INLIER_RATIO and
                self.last_feature_count >= self.THRESH_MIN_FEATURES
            ),
            "dead_reckoning": self.dead_reckoning_active,
            "dr_frame_count": self.dr_frame_count,
            "inlier_ratio": round(self.last_inlier_ratio, 3),
            "feature_count": self.last_feature_count,
            "heading_deg": round(heading_deg, 1) if heading_deg is not None else None,
            "accum_yaw_deg": round(math.degrees(self.accumulated_dr_yaw), 1),
            "scale_learned": self.gps_motion_total >= self.SCALE_LEARN_MIN_DIST,
            "chirality": int(self.chirality),
        }


if __name__ == "__main__":
    vo = MonocularVO()
    d1 = np.zeros((480, 640), dtype=np.uint8)
    cv2.rectangle(d1, (100, 100), (200, 200), 255, -1)
    cv2.rectangle(d1, (300, 300), (400, 400), 255, -1)
    d2 = np.zeros((480, 640), dtype=np.uint8)
    cv2.rectangle(d2, (105, 105), (205, 205), 255, -1)
    cv2.rectangle(d2, (305, 305), (405, 405), 255, -1)
    j1 = {"translation_x": 0.0, "translation_y": 0.0, "translation_z": 10.0, "gps_health_status": 1}
    j2 = {"translation_x": 0.5, "translation_y": 0.0, "translation_z": 10.0, "gps_health_status": 1}
    print(vo.process_frame(d1, j1))
    print(vo.process_frame(d2, j2))
