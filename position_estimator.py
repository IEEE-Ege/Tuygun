import cv2
import numpy as np
import requests
import pandas as pd

CAMERA_PROFILES = {
    "thermal":   {"fx": 731.7965, "fy": 732.0172, "cx": 319.2367, "cy": 251.2424,
                  "k1": -0.3507,  "k2": 0.1137,   "p1": 0.0, "p2": 0.0, "width": 640,  "height": 512},
    "rgb_4k":    {"fx": 2792.2,   "fy": 2795.2,   "cx": 1988.0,   "cy": 1562.2,
                  "k1": 0.0798,   "k2": -0.1867,  "p1": 0.0, "p2": 0.0, "width": 4000, "height": 3000},
    "rgb_1080p": {"fx": 1389.7,   "fy": 1387.1,   "cx": 954.007,  "cy": 558.896,
                  "k1": 0.1378,   "k2": -0.2564,  "p1": 0.0, "p2": 0.0, "width": 1920, "height": 1080},
}

def detect_camera_profile(frame):
    h, w = frame.shape[:2]
    if w == 640 and h == 512: return CAMERA_PROFILES["thermal"]
    elif w >= 3000:            return CAMERA_PROFILES["rgb_4k"]
    return CAMERA_PROFILES["rgb_1080p"]


class PositionEstimator:
    """
    Düzeltmeler (önceki versiyona göre):

    1. Yön offseti: mean(gps) - mean(flow) yerine
       her karede (gps_ang - flow_ang) farkını hesapla,
       bunların döngüsel ortalamasını al → drone viraj yapsa bile
       offset sabit kalır, sürüklenme olmaz.

    2. Scale: affine tx/ty centered koordinatta hesaplanıyor,
       bu raw piksel kaymasından farklı. Düzeltme için
       scale_buf'a (gps_dist / flow_dist_px_raw) kaydediyoruz
       ve GPS bozulunca raw piksel → metre dönüşümünü
       (px / fx) * altitude * scale_correction ile yapıyoruz.

    3. Yumuşak GPS sync korundu.
    """

    SYNC_FRAMES = 15

    def __init__(self, camera_profile=None, assumed_altitude=50.0):
        self.cam              = camera_profile
        self.assumed_altitude = assumed_altitude
        self.prev_gray        = None
        self.prev_kp          = None
        self.estimated_pos    = np.zeros(3)
        self.last_good_pos    = np.zeros(3)
        self.last_known_z     = 0.0
        self.prev_gps         = None
        self.frame_count      = 0

        # Yön: her karedeki (gps_yönü - flow_yönü) farkının döngüsel ortalaması
        self._offset_sin = []   # sin(gps_ang - flow_ang)
        self._offset_cos = []   # cos(gps_ang - flow_ang)
        self._world_angle_offset = 0.0  # öğrenilen sabit offset

        # Scale: gps_dist / flow_dist (piksel cinsinden ham)
        # flow_dist_px: undistort edilmiş, centered olmayan piksel mesafesi
        self._scale_buf    = []
        self._global_scale = 1.0   # metre/piksel
        self.MIN_SAMPLES   = 15

        # GPS sync
        self._sync_active = False
        self._sync_frame  = 0
        self._sync_start  = None
        self._sync_target = None

        self.lk_params = dict(
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        self.feat_params = dict(
            maxCorners=600, qualityLevel=0.005, minDistance=7, blockSize=7)
        self.KP_MIN = 40

    # ── Yardımcılar ────────────────────────────────────────────────────────────

    def _preprocess(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame.copy()
        return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

    def _undistort(self, pts, cam):
        K = np.array([[cam["fx"],0,cam["cx"]],[0,cam["fy"],cam["cy"]],[0,0,1]], np.float64)
        D = np.array([cam["k1"],cam["k2"],cam["p1"],cam["p2"]], np.float64)
        return cv2.undistortPoints(
            pts.astype(np.float32).reshape(-1,1,2), K, D, P=K).reshape(-1, 2)

    def _grid_rejection(self, pp, cp, width, height, grid_size=3):
        flow_vecs = cp - pp
        valid = []
        cw, ch = width / grid_size, height / grid_size
        for i in range(grid_size):
            for j in range(grid_size):
                mask = ((pp[:,0] >= j*cw) & (pp[:,0] < (j+1)*cw) &
                        (pp[:,1] >= i*ch) & (pp[:,1] < (i+1)*ch))
                idx = np.where(mask)[0]
                if len(idx) > 4:
                    lf  = flow_vecs[idx]
                    lm  = np.median(lf, axis=0)
                    d   = np.linalg.norm(lf - lm, axis=1)
                    thr = max(2.0, 1.5 * np.median(d))
                    valid.extend(idx[d < thr])
                else:
                    valid.extend(idx)
        return np.array(valid, dtype=int)

    # ── Optical Flow ───────────────────────────────────────────────────────────

    def _flow_delta(self, prev_gray, curr_gray, cam):
        """
        Döner: (tx_px, ty_px, raw_dx_px, raw_dy_px, n_inlier)
        tx_px / ty_px: affine öteleme (centered koordinat)
        raw_dx_px / raw_dy_px: undistort edilmiş ham piksel kayması (median)
        """
        if self.prev_kp is None or len(self.prev_kp) < self.KP_MIN:
            self.prev_kp = cv2.goodFeaturesToTrack(prev_gray, **self.feat_params)
        if self.prev_kp is None:
            return 0.0, 0.0, 0.0, 0.0, 0

        curr_pts, st, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, self.prev_kp, None, **self.lk_params)
        ok = st.ravel() == 1
        if ok.sum() < 10:
            self.prev_kp = cv2.goodFeaturesToTrack(curr_gray, **self.feat_params)
            return 0.0, 0.0, 0.0, 0.0, 0

        pp = self._undistort(self.prev_kp.reshape(-1, 2)[ok], cam)
        cp = self._undistort(curr_pts.reshape(-1, 2)[ok], cam)

        # Ham piksel kayması (scale kalibrasyonu için)
        raw_flow = cp - pp
        med_raw  = np.median(raw_flow, axis=0)

        # Grid temizleme
        vi = self._grid_rejection(pp, cp, cam["width"], cam["height"])
        if len(vi) < 5:
            self.prev_kp = cv2.goodFeaturesToTrack(curr_gray, **self.feat_params)
            return 0.0, 0.0, float(med_raw[0]), float(med_raw[1]), 0

        # Centered affine (yön için)
        cx, cy = cam["cx"], cam["cy"]
        pp_c = pp[vi] - np.array([cx, cy])
        cp_c = cp[vi] - np.array([cx, cy])

        M, inliers = cv2.estimateAffinePartial2D(
            pp_c, cp_c, method=cv2.RANSAC, ransacReprojThreshold=3.0)

        if M is None or inliers is None or inliers.sum() < 5:
            self.prev_kp = cv2.goodFeaturesToTrack(curr_gray, **self.feat_params)
            return 0.0, 0.0, float(med_raw[0]), float(med_raw[1]), 0

        tx   = float(M[0, 2])
        ty   = float(M[1, 2])
        n_inl = int(inliers.sum())

        if abs(tx) < 0.3: tx = 0.0
        if abs(ty) < 0.3: ty = 0.0

        inl_mask = inliers.ravel().astype(bool)
        kept = curr_pts.reshape(-1, 2)[ok][vi][inl_mask]
        self.prev_kp = kept.reshape(-1, 1, 2) if len(kept) >= self.KP_MIN else None

        return tx, ty, float(med_raw[0]), float(med_raw[1]), n_inl

    # ── Kalibrasyon ────────────────────────────────────────────────────────────

    def _update_calibration(self, tx_px, ty_px, raw_dx, raw_dy, gps_delta):
        """
        Doğru yön öğrenme:
          Her karedeki offset = gps_yönü - flow_yönü
          Bunların döngüsel ortalaması = sabit world_angle_offset

        Doğru scale öğrenme:
          gps_dist / raw_piksel_dist * fx / altitude ≈ 1 olmalı
          Biz direkt: scale = gps_dist / raw_piksel_dist (metre/piksel)
        """
        gps_dist  = float(np.linalg.norm(gps_delta[:2]))
        raw_dist  = float(np.hypot(raw_dx, raw_dy))
        flow_dist = float(np.hypot(tx_px, ty_px))   # yön için

        if gps_dist < 0.05: return   # çok küçük hareket

        # ── Yön: per-frame offset döngüsel ortalaması ──────────────────────
        if flow_dist > 0.5:
            gps_ang  = float(np.arctan2(gps_delta[1], gps_delta[0]))
            # flow: görüntüde noktalar hangi yönde kaydı → kamera hareketi tersine
            flow_ang = float(np.arctan2(-ty_px, -tx_px))
            offset   = gps_ang - flow_ang

            self._offset_sin.append(float(np.sin(offset)))
            self._offset_cos.append(float(np.cos(offset)))
            if len(self._offset_sin) > 300:
                self._offset_sin.pop(0)
                self._offset_cos.pop(0)

            if len(self._offset_sin) >= self.MIN_SAMPLES:
                # Döngüsel ortalama: per-frame farkların ortalaması
                self._world_angle_offset = float(np.arctan2(
                    np.mean(self._offset_sin),
                    np.mean(self._offset_cos)))

        # ── Scale: gps_dist / raw_piksel_dist ─────────────────────────────
        if raw_dist > 1.0:   # en az 1 piksel ham hareket
            scale = gps_dist / raw_dist   # metre/piksel
            # Makul aralık: 0.001 – 1.0 m/piksel
            if 0.001 < scale < 1.0:
                self._scale_buf.append(scale)
                if len(self._scale_buf) > 300:
                    self._scale_buf.pop(0)
                if len(self._scale_buf) >= self.MIN_SAMPLES:
                    arr = np.array(self._scale_buf)
                    med = np.median(arr)
                    inl = np.abs(arr - med) < 2.0 * np.std(arr)
                    if inl.sum() > 5:
                        self._global_scale = float(np.median(arr[inl]))

    # ── Ana İşlem ─────────────────────────────────────────────────────────────

    def process(self, frame, gps_healthy, server_x, server_y, server_z):
        cam       = self.cam if self.cam else detect_camera_profile(frame)
        curr_gray = self._preprocess(frame)
        self.frame_count += 1
        gps_pos   = np.array([server_x, server_y, server_z])

        if self.prev_gray is not None:
            tx_px, ty_px, raw_dx, raw_dy, n_inl = self._flow_delta(
                self.prev_gray, curr_gray, cam)
        else:
            tx_px, ty_px, raw_dx, raw_dy, n_inl = 0.0, 0.0, 0.0, 0.0, 0

        if gps_healthy:
            # Kalibrasyon öğren
            if self.prev_gps is not None and n_inl >= 5:
                self._update_calibration(
                    tx_px, ty_px, raw_dx, raw_dy, gps_pos - self.prev_gps)

            # Sync aktifse kademeli yaklaş
            if self._sync_active:
                self._sync_frame += 1
                t = min(self._sync_frame / self.SYNC_FRAMES, 1.0)
                self.estimated_pos = (1-t) * self._sync_start + t * self._sync_target
                if self._sync_frame >= self.SYNC_FRAMES:
                    self._sync_active  = False
                    self.estimated_pos = gps_pos.copy()
            else:
                self.estimated_pos = gps_pos.copy()

            self.last_good_pos = gps_pos.copy()
            self.last_known_z  = server_z
            self.prev_gps      = gps_pos.copy()

            # Drift kontrolü: GPS geri geldi mi, büyük sapma var mı?
            drift = float(np.linalg.norm(self.estimated_pos[:2] - gps_pos[:2]))
            if drift > 2.0 and not self._sync_active:
                self._sync_active = True
                self._sync_frame  = 0
                self._sync_start  = self.estimated_pos.copy()
                self._sync_target = gps_pos.copy()

            result = tuple(float(v) for v in self.estimated_pos)

        else:
            # GPS bozuk: raw piksel kayması → ölçek → yön döndürme
            if n_inl >= 5 and abs(raw_dx) + abs(raw_dy) > 0.1:
                # Ham piksel → metre (scale = metre/piksel olarak öğrenildi)
                dx_m = (-raw_dx) * self._global_scale
                dy_m = (-raw_dy) * self._global_scale

                # Dünya yönüne döndür
                a = self._world_angle_offset
                world_dx = dx_m * np.cos(a) - dy_m * np.sin(a)
                world_dy = dx_m * np.sin(a) + dy_m * np.cos(a)

                self.estimated_pos[0] += world_dx
                self.estimated_pos[1] += world_dy
                self.estimated_pos[2]  = self.last_known_z

            self.prev_gps = None
            result = tuple(float(v) for v in self.estimated_pos)

        self.prev_gray = curr_gray
        return result

    @property
    def scale(self): return self._global_scale

    @property
    def n_scale_samples(self): return len(self._scale_buf)

    @property
    def world_angle_deg(self): return float(np.degrees(self._world_angle_offset))


# ─── Sunucu Yardımcıları ──────────────────────────────────────────────────────

def fetch_frame_list(server_url):
    r = requests.get(f"{server_url}/frames/", timeout=10)
    r.raise_for_status()
    return r.json()

def download_image(image_url, server_url):
    url = image_url if image_url.startswith("http") else server_url + image_url
    r   = requests.get(url, timeout=10)
    arr = np.frombuffer(r.content, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def send_result(server_url, user_url, frame_ref, tx, ty, tz,
                detected_objects=None, detected_undefined_objects=None):
    payload = {
        "user":  user_url, "frame": frame_ref,
        "detected_objects": detected_objects or [],
        "detected_translations": [
            {"translation_x": tx, "translation_y": ty, "translation_z": tz}],
        "detected_undefined_objects": detected_undefined_objects or [],
    }
    return requests.post(f"{server_url}/results/", json=payload, timeout=10).status_code

def run_competition(server_url, user_url, camera_profile=None, assumed_altitude=50.0):
    estimator = PositionEstimator(camera_profile, assumed_altitude)
    frames    = fetch_frame_list(server_url)
    print(f"Toplam {len(frames)} kare.")
    for i, fd in enumerate(frames):
        image = download_image(fd["image_url"], server_url)
        tx, ty, tz = estimator.process(
            image, fd.get("gps_health_status", 1) == 1,
            fd.get("translation_x", 0.0),
            fd.get("translation_y", 0.0),
            fd.get("translation_z", 0.0))
        status = send_result(server_url, user_url, fd["url"], tx, ty, tz)
        print(f"[{i+1}/{len(frames)}] x:{tx:8.3f} y:{ty:8.3f} z:{tz:8.3f} "
              f"scale:{estimator.scale:.5f} angle:{estimator.world_angle_deg:.1f}deg HTTP:{status}")

if __name__ == "__main__":
    run_competition("http://127.0.0.25:5000", "http://localhost/users/4/")