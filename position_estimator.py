import cv2
import numpy as np
import requests
import pandas as pd

# ─── Kamera Profilleri ────────────────────────────────────────────────────────

CAMERA_PROFILES = {
    "thermal": {
        "fx": 731.7965, "fy": 732.0172,
        "cx": 319.2367, "cy": 251.2424,
        "k1": -0.3507,  "k2": 0.1137,
        "p1": 0.0,      "p2": 0.0,
        "width": 640,   "height": 512,
    },
    "rgb_4k": {
        "fx": 2792.2, "fy": 2795.2,
        "cx": 1988.0, "cy": 1562.2,
        "k1": 0.0798, "k2": -0.1867,
        "p1": 0.0,    "p2": 0.0,
        "width": 4000, "height": 3000,
    },
    "rgb_1080p": {
        "fx": 1389.7, "fy": 1387.1,
        "cx": 954.007, "cy": 558.896,
        "k1": 0.1378, "k2": -0.2564,
        "p1": 0.0,    "p2": 0.0,
        "width": 1920, "height": 1080,
    },
}

def detect_camera_profile(frame):
    h, w = frame.shape[:2]
    if w == 640 and h == 512:
        return CAMERA_PROFILES["thermal"]
    elif w >= 3000:
        return CAMERA_PROFILES["rgb_4k"]
    return CAMERA_PROFILES["rgb_1080p"]


# ─── Pozisyon Tahmincisi ──────────────────────────────────────────────────────

class PositionEstimator:
    def __init__(self, camera_profile=None, assumed_altitude=50.0):
        self.cam              = camera_profile
        # Gerçek irtifa bilinmiyorsa sabit değer kullan.
        # Yarışmada telemetri paylaşılırsa burası güncellenir.
        self.assumed_altitude = assumed_altitude

        # Durum
        self.prev_gray      = None
        self.prev_kp        = None
        self.estimated_pos  = np.zeros(3)
        self.last_good_pos  = np.zeros(3)
        self.prev_gps       = None
        self.frame_count    = 0

        # Scale öğrenme: GPS sağlıklıyken flow/GPS oranını öğren
        # Ayrı X ve Y yerine tek bir isotropic scale kullan (daha stabil)
        self._scale_buf  = []          # (flow_mag, gps_mag) çiftleri
        self._scale      = 1.0         # öğrenilen ölçek
        self.MIN_SAMPLES = 20

        # Lucas-Kanade parametreleri
        self.lk_params = dict(
            winSize  = (21, 21),
            maxLevel = 3,
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        self.feat_params = dict(
            maxCorners   = 500,
            qualityLevel = 0.005,   # daha fazla nokta bul
            minDistance  = 7,
            blockSize    = 7,
        )
        self.KP_MIN_THRESHOLD = 20   # bu altına düşünce keypoint yenile

    # ── Yardımcılar ────────────────────────────────────────────────────────────

    def _preprocess(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame.copy()
        return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

    def _undistort(self, pts, cam):
        K = np.array([[cam["fx"],0,cam["cx"]],[0,cam["fy"],cam["cy"]],[0,0,1]], np.float64)
        D = np.array([cam["k1"],cam["k2"],cam["p1"],cam["p2"]], np.float64)
        return cv2.undistortPoints(
            pts.astype(np.float32).reshape(-1,1,2), K, D, P=K
        ).reshape(-1, 2)

    def _refresh_kp(self, gray):
        kp = cv2.goodFeaturesToTrack(gray, **self.feat_params)
        self.prev_kp = kp

    # ── Optical Flow Delta ────────────────────────────────────────────────────

    def _flow_delta(self, prev_gray, curr_gray, cam):
        """
        İki kare arasındaki piksel kaymasını hesapla.
        Döndür: (dx_px, dy_px, n_inlier) — ham piksel cinsinden
        """
        if self.prev_kp is None or len(self.prev_kp) < self.KP_MIN_THRESHOLD:
            self._refresh_kp(prev_gray)

        if self.prev_kp is None:
            return 0.0, 0.0, 0

        curr_pts, st, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, self.prev_kp, None, **self.lk_params)

        ok = st.ravel() == 1
        if ok.sum() < 5:
            self._refresh_kp(curr_gray)
            return 0.0, 0.0, 0

        pp = self._undistort(self.prev_kp.reshape(-1,2)[ok], cam)
        cp = self._undistort(curr_pts.reshape(-1,2)[ok], cam)
        flow = cp - pp   # pozitif = nokta sağa/aşağı kaydı

        # RANSAC benzeri outlier temizleme
        med  = np.median(flow, axis=0)
        dist = np.linalg.norm(flow - med, axis=1)
        thr  = np.percentile(dist, 70)
        inl  = dist < thr
        final = np.median(flow[inl], axis=0) if inl.sum() >= 5 else med
        n_inl = int(inl.sum())

        # Keypoint'leri güncelle — zorla yenilemek yerine sadece yetersiz olunca yenile
        self.prev_kp = curr_pts.reshape(-1,2)[ok][inl].reshape(-1,1,2) if inl.sum() >= self.KP_MIN_THRESHOLD else None

        return float(final[0]), float(final[1]), n_inl

    # ── Scale Öğrenme ─────────────────────────────────────────────────────────

    def _learn_scale(self, dx_px, dy_px, cam, gps_delta, altitude):
        """
        GPS sağlıklıyken:
          flow_m = (px / fx) * altitude
          scale  = gps_m / flow_m
        """
        gps_dist = float(np.linalg.norm(gps_delta[:2]))
        if gps_dist < 0.05:
            return  # çok küçük hareket → gürültü

        flow_dx_m = -(dx_px / cam["fx"]) * altitude
        flow_dy_m = -(dy_px / cam["fy"]) * altitude
        flow_dist = float(np.hypot(flow_dx_m, flow_dy_m))

        if flow_dist < 1e-4:
            return

        ratio = gps_dist / flow_dist
        # Mantıksız değerleri at (0.01x - 100x arası)
        if 0.01 < ratio < 100:
            self._scale_buf.append(ratio)

        if len(self._scale_buf) > 300:
            self._scale_buf.pop(0)

        if len(self._scale_buf) >= self.MIN_SAMPLES:
            arr = np.array(self._scale_buf)
            med = np.median(arr)
            inl = np.abs(arr - med) < 2 * arr.std()
            if inl.sum() > 5:
                self._scale = float(np.median(arr[inl]))

    # ── Ana İşlem ─────────────────────────────────────────────────────────────

    def process(self, frame, gps_healthy, server_x, server_y, server_z):
        cam       = self.cam if self.cam else detect_camera_profile(frame)
        curr_gray = self._preprocess(frame)
        self.frame_count += 1
        gps_pos   = np.array([server_x, server_y, server_z])

        # İrtifa: Z offseti çok küçük, sabit varsayım kullan
        # Gerçek irtifa yarışmada telemetri ile gelirse burası güncellenir
        altitude = self.assumed_altitude

        if gps_healthy:
            # Flow HER ZAMAN çalışsın → ısınmış kalsın + scale öğren
            if self.prev_gray is not None:
                dx_px, dy_px, n_inl = self._flow_delta(self.prev_gray, curr_gray, cam)
                if self.prev_gps is not None and n_inl >= 5:
                    gps_delta = gps_pos - self.prev_gps
                    self._learn_scale(dx_px, dy_px, cam, gps_delta, altitude)

            self.estimated_pos = gps_pos.copy()
            self.last_good_pos = gps_pos.copy()
            self.prev_gps      = gps_pos.copy()
            result = (server_x, server_y, server_z)

        else:
            # GPS bozuk: scale uygulanmış flow ile tahmin et
            if self.prev_gray is not None:
                dx_px, dy_px, n_inl = self._flow_delta(self.prev_gray, curr_gray, cam)
                if n_inl >= 5:
                    # Piksel → metre → scale düzeltmesi
                    dx_m = -(dx_px / cam["fx"]) * altitude * self._scale
                    dy_m = -(dy_px / cam["fy"]) * altitude * self._scale
                    self.estimated_pos[0] += dx_m
                    self.estimated_pos[1] += dy_m
                    # Z: flow ile tahmin edilemiyor, son bilinen değerde tut

            self.prev_gps = None
            result = tuple(float(v) for v in self.estimated_pos)

        self.prev_gray = curr_gray
        return result

    @property
    def scale(self):
        return self._scale

    @property
    def n_scale_samples(self):
        return len(self._scale_buf)


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
        "user":  user_url,
        "frame": frame_ref,
        "detected_objects": detected_objects or [],
        "detected_translations": [
            {"translation_x": tx, "translation_y": ty, "translation_z": tz}
        ],
        "detected_undefined_objects": detected_undefined_objects or [],
    }
    return requests.post(f"{server_url}/results/", json=payload, timeout=10).status_code


# ─── Ana Yarışma Döngüsü ──────────────────────────────────────────────────────

def run_competition(server_url, user_url, camera_profile=None, assumed_altitude=50.0):
    estimator = PositionEstimator(camera_profile, assumed_altitude)
    frames    = fetch_frame_list(server_url)
    print(f"Toplam {len(frames)} kare.")
    for i, fd in enumerate(frames):
        image = download_image(fd["image_url"], server_url)
        tx, ty, tz = estimator.process(
            image,
            fd.get("gps_health_status", 1) == 1,
            fd.get("translation_x", 0.0),
            fd.get("translation_y", 0.0),
            fd.get("translation_z", 0.0),
        )
        status = send_result(server_url, user_url, fd["url"], tx, ty, tz)
        print(f"[{i+1}/{len(frames)}] x:{tx:8.3f} y:{ty:8.3f} z:{tz:8.3f} "
              f"scale:{estimator.scale:.3f} HTTP:{status}")


if __name__ == "__main__":
    SERVER_URL = "http://127.0.0.25:5000"
    USER_URL   = "http://localhost/users/4/"
    CAMERA     = None          # None → otomatik algıla
    ALTITUDE   = 50.0          # metre — yarışmada güncelle
    run_competition(SERVER_URL, USER_URL, CAMERA, ALTITUDE)