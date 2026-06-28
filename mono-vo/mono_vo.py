import cv2
import numpy as np
import math

class MonocularVO:
    def __init__(self, min_num_feat=2000):
        self.min_num_feat = min_num_feat
        
        # Görev 3: K Matrisi - İHA kamerasının kalibrasyon matrisi (manuel girilecek boş array)
        self.K = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        
        # Kamera odak uzaklığı ve optik merkez (K matrisinden alınacak, K doldurulduğunda güncellenmeli)
        # Örnek değerler, gerçek K matrisi ile ezilmelidir. (C++ kodundan varsayılan alınmıştır)
        self.focal = 718.8560
        self.pp = (607.1928, 185.2157)
        
        # Değişkenler
        self.px_ref = None  # Önceki karenin (frame i-1) özellik noktaları
        self.px_cur = None  # Geçerli karenin (frame i) özellik noktaları
        self.prev_frame = None
        self.cur_frame = None
        
        # Global Pose matrisleri (3x3 Rotasyon, 3x1 Translation)
        self.R_f = np.eye(3, dtype=np.float64)
        self.t_f = np.zeros((3, 1), dtype=np.float64)
        
        # GPS/Telemetri koptuğunda (gps_health_status=0) dead-reckoning yapabilmek için
        self.last_valid_scale = 1.0
        self.last_valid_velocity = 0.0  # metre/kare (son bilinen hız)
        self.frames_since_keyframe = 0  # Keyframe'den bu yana geçen kare sayısı
        self.pixel_to_meter = 0.01  # piksel->metre dönüşüm oranı (GPS'ten öğrenilecek)
        
        # JSON verisini takip için
        self.prev_json_data = None
        self.is_initialized = False
        
        # Feature detector (FAST) - Görev 1
        self.detector = cv2.FastFeatureDetector_create(threshold=20, nonmaxSuppression=True)
        
        # KLT parameters
        self.lk_params = dict(winSize=(21, 21),
                              maxLevel=4, # Hızlı dönüşlerde (Yaw) özellikleri kaybetmemek için piramit seviyesi artırıldı
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    def update_calibration(self, focal, pp):
        """K matrisinden hesaplanan kalibrasyon değerlerini günceller."""
        self.focal = focal
        self.pp = pp

    def feature_detection(self, img):
        """FAST kullanarak özellik noktalarını tespit eder."""
        keypoints = self.detector.detect(img, None)
        if not keypoints:
            return np.empty((0, 2), dtype=np.float32)
        # Sadece x,y koordinatlarını numpy dizisi olarak döndür
        return np.array([kp.pt for kp in keypoints], dtype=np.float32)

    def feature_tracking(self, img_ref, img_cur, px_ref):
        """KLT kullanarak bir kareden diğerine özellikleri takip eder."""
        kp2, st, err = cv2.calcOpticalFlowPyrLK(img_ref, img_cur, px_ref, None, **self.lk_params)
        
        # Sadece başarıyla takip edilen noktaları filtrele
        st = st.reshape(st.shape[0])
        kp1 = px_ref[st == 1]
        kp2 = kp2[st == 1]
        
        return kp1, kp2

    def get_absolute_scale(self, cur_json, keyframe_json, pixel_displacement=0.0):
        """
        Görev 2: Scale (Ölçek) Enjeksiyonu
        GPS sağlıklıysa: Keyframe'den mevcut kareye kadar olan gerçek mesafeyi döner
                          ve piksel→metre oranını (PPM) öğrenir.
        GPS kesilmişse: Öğrenilmiş piksel→metre oranını kullanarak piksel kaymasından 
                        gerçek mesafeyi tahmin eder.
        """
        if cur_json.get("gps_health_status", 0) == 1 and keyframe_json is not None:
            # GPS SAĞLIKLI: Keyframe'den şimdiye kadar olan gerçek mesafe
            x_prev = keyframe_json.get("translation_x", 0.0)
            y_prev = keyframe_json.get("translation_y", 0.0)
            z_prev = keyframe_json.get("translation_z", 0.0)
            
            x_cur = cur_json.get("translation_x", 0.0)
            y_cur = cur_json.get("translation_y", 0.0)
            z_cur = cur_json.get("translation_z", 0.0)
            
            scale = math.sqrt((x_cur - x_prev)**2 + (y_cur - y_prev)**2 + (z_cur - z_prev)**2)
            
            # Piksel→Metre oranını öğren (PPM = Pixel Per Meter)
            # Bu oran kameranın irtifası ve odak uzaklığına bağlıdır.
            # Daha yüksek hızlı karelerden öğrenmek, oranı daha kesin yapar (scale error'u önler).
            if pixel_displacement > 2.0 and scale > 0.05:
                self.pixel_to_meter = scale / pixel_displacement
            
            self.last_valid_scale = scale
            return scale
        else:
            # GPS KESİLDİ: Dead-reckoning — Piksel kayması × Öğrenilmiş piksel→metre oranı
            estimated_scale = pixel_displacement * self.pixel_to_meter
            return max(estimated_scale, 0.0)

    def process_frame(self, image, json_data):
        """
        Her bir yeni kare ve JSON verisi için çalıştırılacak ana metot.
        image: Gri seviyeye (grayscale) dönüştürülmüş OpenCV resmi.
        json_data: {"translation_x": x, "translation_y": y, "translation_z": z, "gps_health_status": status} formatında dict.
        """
        self.cur_frame = image
        
        # Her kare i\u00e7in sayac\u0131 art\u0131r (init'te s\u0131f\u0131rlan\u0131r, keyframe'de s\u0131f\u0131rlan\u0131r)
        self.frames_since_keyframe += 1
        
        if not self.is_initialized:
            # İlk kare: sadece özellikleri tespit et ve bekle
            self.px_ref = self.feature_detection(self.cur_frame)
            self.prev_frame = self.cur_frame.copy()
            self.prev_json_data = json_data
            self.keyframe_json = json_data  # Keyframe anındaki GPS pozisyonu
            self.is_initialized = True
            return self._format_output()
            
        if len(self.px_ref) == 0:
            # Eğer referans noktası yoksa yeniden tespit et
            self.px_ref = self.feature_detection(self.prev_frame)
            if len(self.px_ref) == 0:
                self.prev_frame = self.cur_frame.copy()
                self.prev_json_data = json_data
                self.keyframe_json = json_data
                return self._format_output()
            
        # Özellikleri takip et (Referans kare ile mevcut kare arasında)
        self.px_ref, self.px_cur = self.feature_tracking(self.prev_frame, self.cur_frame, self.px_ref)
        
        # Eğer yeterli özellik yoksa, yeniden tespiti zorla
        if len(self.px_ref) < 8:
             self.px_cur = self.feature_detection(self.cur_frame)
             self.px_ref = self.px_cur
             self.prev_frame = self.cur_frame.copy()
             self.prev_json_data = json_data
             self.keyframe_json = json_data
             return self._format_output()

        # Geleneksel Essential Matris hesaplaması aşağı bakan kamerada yeryüzü düzlemsel olduğu için yozlaşır.
        # ÇÖZÜM: İHA aşağı bakarken, gimbalın stabilize olduğu varsayımıyla, piksellerin medyan dönüş (Yaw) 
        # ve medyan öteleme miktarı (Translation) bize doğrudan kamera hareketini verir.
        
        # Affine dönüşüm: dönüşü ve ötelemeyi ayrıştırma (merkeze göre)
        px_cur_c = self.px_cur - np.array(self.pp)
        px_ref_c = self.px_ref - np.array(self.pp)
        
        m, inliers = cv2.estimateAffinePartial2D(
            px_cur_c, px_ref_c,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
            maxIters=2000,
            confidence=0.99
        )
        
        # Minimum piksel kayması eşiği: bu eşiğin altındaki kaymalar gürültüdür.
        # Referans kareyi GÜNCELLEMEYİP bir sonraki kareye geçeriz.
        # Bu sayede yavaş uçuşlarda piksel kayması birden fazla kare boyunca birikir
        # ve gürültü seviyesinden çıkarak ölçülebilir hale gelir. (Adaptive Keyframe)
        MIN_PIXEL_DISP = 0.5  # piksel
        
        if m is not None:
            dx = m[0, 2]
            dy = m[1, 2]
            pixel_displacement = np.sqrt(dx**2 + dy**2)
            
            if pixel_displacement < MIN_PIXEL_DISP:
                # Yeterli hareket yok — referans kareyi DEĞİŞTİRME, sadece json güncelle.
                # prev_frame ve px_ref aynı kalır → bir sonraki karede fark daha büyük olacak.
                self.prev_json_data = json_data
                return self._format_output()
            
            # Yeterli hareket var — Keyframe güncelle ve pozisyonu hesapla.
            yaw = -np.arctan2(m[1, 0], m[0, 0])
            
            # Öteleme vektörünü normalize et (yön bilgisi)
            t_new = np.array([[dx], [dy], [0.0]], dtype=np.float64)
            t_new = t_new / pixel_displacement  # Birim vektör
                
            # Kameranın Z ekseni (Yaw) etrafındaki dönüşü
            R_new = np.array([
                [np.cos(yaw), -np.sin(yaw), 0],
                [np.sin(yaw),  np.cos(yaw), 0],
                [0,            0,           1]
            ], dtype=np.float64)
        else:
            self.prev_json_data = json_data
            return self._format_output()
        
        # Ölçek (Scale) hesaplama — Keyframe'den bu kareye kadar olan toplam GPS mesafesi
        scale = self.get_absolute_scale(json_data, self.keyframe_json, pixel_displacement)
        
        # Hareket varsa global pose matrislerini güncelle
        if scale > 0.001:
            self.t_f = self.t_f + scale * self.R_f.dot(t_new)
            self.R_f = R_new.dot(self.R_f)
            
        # Özellik sayısı belirli bir eşiğin altına düşerse yeniden tespit (Redetection)
        if self.px_ref.shape[0] < self.min_num_feat:
            self.px_cur = self.feature_detection(self.cur_frame)
            
        # KEYFRAME GÜNCELLEMESİ: Artık yeterli hareket algılandı, yeni referans noktası oluştur
        self.prev_frame = self.cur_frame.copy()
        self.px_ref = self.px_cur
        self.prev_json_data = json_data
        self.keyframe_json = json_data  # Bu kare artık yeni keyframe
        self.frames_since_keyframe = 0  # Sayacı sıfırla
        
        return self._format_output()

    def _format_output(self):
        """
        Görev 4: Çıktı Formatı & Görev 3: Eksen Dönüşümü
        Aşağı bakan (yeryüzüne) kamera modeli ve CSV Ground Truth Eksenleri:
        - İleri gidiş (CSV translation_y): estimateAffinePartial2D'den dönen dy negatif olur. 
          t_f[1][0] negatifleşir. Bu yüzden CSV'ye uydurmak için translation_y = -t_f[1][0]
        - Sağa gidiş (CSV translation_x): dx negatif olur. 
          t_f[0][0] negatifleşir. Bu yüzden CSV'ye uydurmak için translation_x = -t_f[0][0]
        """
        # CSV ile birebir eşleşmesi için eksenleri ayarlıyoruz:
        x_val = float(self.t_f[0][0])   # Sağa / Sola (kullanıcı talebiyle - ile çarpıldı)
        y_val = float(-self.t_f[1][0])  # İleri / Geri
        z_val = float(self.t_f[2][0])
        
        output = {
            "detected_translations": [
                {
                    "translation_x": x_val,
                    "translation_y": y_val,
                    "translation_z": z_val
                }
            ]
        }
        return output

# --- KULLANIM ÖRNEĞİ (Test için) ---
if __name__ == "__main__":
    vo = MonocularVO()
    
    # Kullanıcı tarafından kalibrasyon matrisi doldurulduğunda:
    # vo.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    # vo.update_calibration(focal=vo.K[0,0], pp=(vo.K[0,2], vo.K[1,2]))
    
    # Simüle edilmiş (Dummy) veriler - FAST'ın özellik bulabilmesi için dikdörtgenler eklendi
    dummy_img1 = np.zeros((480, 640), dtype=np.uint8)
    cv2.rectangle(dummy_img1, (100, 100), (200, 200), 255, -1)
    cv2.rectangle(dummy_img1, (300, 300), (400, 400), 255, -1)
    
    dummy_img2 = np.zeros((480, 640), dtype=np.uint8)
    cv2.rectangle(dummy_img2, (105, 105), (205, 205), 255, -1)
    cv2.rectangle(dummy_img2, (305, 305), (405, 405), 255, -1)
    
    # Başlangıç JSON verisi
    dummy_json1 = {"translation_x": 0.0, "translation_y": 0.0, "translation_z": 10.0, "gps_health_status": 1}
    # Sonraki kare JSON verisi (örneğin X ekseninde 0.5 metre hareket edilmiş)
    dummy_json2 = {"translation_x": 0.5, "translation_y": 0.0, "translation_z": 10.0, "gps_health_status": 1}
    
    print("Frame 1 İşleniyor...")
    out1 = vo.process_frame(dummy_img1, dummy_json1)
    print("Çıktı 1:", out1)
    
    print("\nFrame 2 İşleniyor...")
    out2 = vo.process_frame(dummy_img2, dummy_json2)
    print("Çıktı 2:", out2)
