import cv2
import numpy as np
import math


class MonocularVO:
    def __init__(self, min_num_feat=2000, sensor_type="RGB"):
        self.min_num_feat = min_num_feat
        self.sensor_type = sensor_type

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
        self.heading_angle = None
        self.heading_initialized = False
        self.heading_alpha = 0.15         # EMA ağırlığı (GPS güncellemesi)
        self.heading_gps_min_dist = 0.05  # GPS deltası bu m'nin altındaysa atla
        self.heading_pix_min_disp = 3.0   # piksel delta bu px'in altındaysa atla

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
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)) if self.sensor_type == "THERMAL" else None

        # ── Detektör ve Parametreler ─────────────────────────────────────────────
        fast_threshold = 10 if self.sensor_type == "THERMAL" else 20
        self.detector = cv2.FastFeatureDetector_create(threshold=fast_threshold, nonmaxSuppression=True)
        self.lk_params = dict(winSize=(21, 21),
                              maxLevel=5,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

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
        
        if self.sensor_type == "THERMAL":
            # Termal kameralarda ortadaki sabit crosshair'i (hedef imlecini) maskele
            cy, cx = h // 2, w // 2
            mask[cy-40:cy+40, cx-40:cx+40] = 0
            
            # Termal için gürültüye daha dayanıklı Shi-Tomasi kullan
            corners = cv2.goodFeaturesToTrack(img, maxCorners=3000, qualityLevel=0.01, minDistance=5, mask=mask)
            if corners is not None:
                return corners.reshape(-1, 2)
            return np.empty((0, 2), dtype=np.float32)

        keypoints = self.detector.detect(img, mask=mask)
        if not keypoints:
            return np.empty((0, 2), dtype=np.float32)
        keypoints = sorted(keypoints, key=lambda kp: kp.response, reverse=True)[:3000]
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
        if (self.keyframe_json is None
                or pixel_displacement < self.heading_pix_min_disp):
            return

        kf_x = self.keyframe_json.get("translation_x", 0.0)
        kf_y = self.keyframe_json.get("translation_y", 0.0)
        gps_dx = json_data.get("translation_x", 0.0) - kf_x
        gps_dy = json_data.get("translation_y", 0.0) - kf_y
        gps_dist = math.sqrt(gps_dx**2 + gps_dy**2)

        if gps_dist < self.heading_gps_min_dist:
            return

        # Dünya üzerindeki gps hareketini kamera frame'ine geri döndür
        phi = self.heading_angle - math.pi / 2
        c, s = math.cos(phi), math.sin(phi)
        
        cx_meter = c * gps_dx + s * gps_dy
        cy_meter = -s * gps_dx + c * gps_dy
        
        cx_pixel = dx
        cy_pixel = -dy

        ppm_general = gps_dist / pixel_displacement

        if self.sensor_type == "THERMAL":
            PPM_X_MULTIPLIER = 0.85
            PPM_Y_MULTIPLIER = 0.85
        else:
            PPM_X_MULTIPLIER = 1.0
            PPM_Y_MULTIPLIER = 1.0

        new_ppm_y = ppm_general * PPM_Y_MULTIPLIER
        new_ppm_x = ppm_general * PPM_X_MULTIPLIER

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
        # theta_pixel : piksel hareket yönü (görüntü y ters olduğu için -dy)
        # heading_angle = theta_world - theta_pixel + π/2
        theta_world = math.atan2(gps_dy, gps_dx)
        theta_pixel = math.atan2(-dy, dx)
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
        # Soft deadband: 0.02 derece altındaki dönüşleri tamamen gürültü (sistematik hata) kabul et
        deadband = math.radians(0.02)
        if abs(yaw) < deadband:
            yaw = 0.0
        else:
            yaw = math.copysign(abs(yaw) - deadband, yaw)
            
        # Fiziksel limit ~2°/kare
        yaw = max(-math.radians(2), min(math.radians(2), yaw))
        self.heading_angle = self._wrap_angle(self.heading_angle - yaw)
        self.accumulated_dr_yaw += abs(yaw)  # heading güvensizliği takibi

        # Kamera frame'indeki metrik hareket:
        dx_meter = dx * self.ppm_x
        dy_meter = -dy * self.ppm_y

        # Dünya birim vektörü hesabını kamera metre hesabıyla yapalım:
        phi = self.heading_angle - math.pi / 2
        c, s = math.cos(phi), math.sin(phi)
        
        world_dx = c * dx_meter - s * dy_meter
        world_dy = s * dx_meter + c * dy_meter

        # Fizik dışı atlamayı sınırla (kalıbozuk optik akış)
        dist = math.sqrt(world_dx**2 + world_dy**2)
        if dist > self.THRESH_MAX_VEL_MF:
            ratio = self.THRESH_MAX_VEL_MF / dist
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
            alpha_scale = 0.0 if self.sensor_type == "THERMAL" else 1.3
            smoothed_scale = 1.0 + (scale - 1.0) * alpha_scale
            self.ppm_x *= smoothed_scale
            self.ppm_y *= smoothed_scale

        # Z ekseni için saf ppm kullanılarak raw hesaplama
        current_z_agl = self.focal * ((self.ppm_x + self.ppm_y) / 2.0)
        # Z Ekseni, dataset'te muhtemelen Aşağı-Pozitif (NED) veya zıt yönlü. AGL ise Yukarı-Pozitif.
        # Bu yüzden AGL'yi eksi (-) ile işleme alıyoruz ki mirror (ayna) efekti düzelsin.
        raw_vo_z = self.z_offset - current_z_agl - self.initial_gps[2]
        
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
                dynamic_multiplier = 1.0 + drift_compensation * 2.0
                corrected_delta_z = delta_z * dynamic_multiplier
            else:
                corrected_delta_z = delta_z * 0.6  # Yükselmeyi / tepeyi küçült (bu iyi çalışıyordu)
                
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
            ransacReprojThreshold=1.5,
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
                confidence >= 0.4 and
                self.last_inlier_ratio >= self.THRESH_INLIER_RATIO and
                self.last_feature_count >= self.THRESH_MIN_FEATURES
            ),
            "dead_reckoning": self.dead_reckoning_active,
            "dr_frame_count": self.dr_frame_count,
            "inlier_ratio": round(self.last_inlier_ratio, 3),
            "feature_count": self.last_feature_count,
            "heading_deg": round(heading_deg, 1) if heading_deg is not None else None,
            "accum_yaw_deg": round(math.degrees(self.accumulated_dr_yaw), 1),
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
