# ═══════════════════════════════════════════════════════════════════════════
#  YANSIMA (CHIRALITY) TEŞHİSİ
#
#  Soru: GPS hareket yönü ile optik akış yönü arasındaki ilişki bir
#  ROTASYON mu (sabit heading yeterli) yoksa YANSIMA mı (eksen aynalı)?
#
#  Matematik: theta_world = GPS hareket açısı, theta_pixel = akış açısı.
#    • Saf rotasyon ise:  A = wrap(theta_world - theta_pixel)  SABİT kalır
#    • Yansıma varsa:     B = wrap(theta_world + theta_pixel)  SABİT kalır
#  Hangisinin dairesel yayılımı küçükse geometri odur.
#
#      python3 araclar/yansima_teshis.py <video> <csv> [process_scale] [max_kare]
# ═══════════════════════════════════════════════════════════════════════════
import math
import sys
import os

import cv2
import numpy as np
import pandas as pd

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def dairesel_yayilim(acilar):
    """Dairesel standart sapma benzeri ölçü: 1 - |ortalama birim vektör| (0=sabit)."""
    c = np.mean(np.cos(acilar))
    s = np.mean(np.sin(acilar))
    R = math.sqrt(c * c + s * s)
    return 1.0 - R, math.degrees(math.atan2(s, c))


def main():
    video, csv = sys.argv[1], sys.argv[2]
    scale = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
    max_kare = int(sys.argv[4]) if len(sys.argv) > 4 else 10**9

    df = pd.read_csv(csv)
    cap = cv2.VideoCapture(video)
    prev = None
    px = None

    A_list, B_list = [], []
    n = min(len(df), max_kare)

    for i in range(n):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if scale != 1.0:
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        if prev is not None and px is not None and len(px) >= 8:
            cur, st, _ = cv2.calcOpticalFlowPyrLK(prev, gray, px, None,
                                                  winSize=(21, 21), maxLevel=5)
            st = st.reshape(-1)
            p0, p1 = px[st == 1], cur[st == 1]
            if len(p0) >= 8:
                m, _ = cv2.estimateAffinePartial2D(p1, p0, method=cv2.RANSAC,
                                                   ransacReprojThreshold=0.6)
                if m is not None:
                    dx, dy = m[0, 2], m[1, 2]
                    pix_disp = math.hypot(dx, dy)
                    gdx = df.translation_x[i] - df.translation_x[i - 1]
                    gdy = df.translation_y[i] - df.translation_y[i - 1]
                    gps_disp = math.hypot(gdx, gdy)
                    if pix_disp > 2.0 and gps_disp > 0.05:
                        tw = math.atan2(gdy, gdx)
                        tp = math.atan2(-dy, dx)   # görüntü y ekseni ters
                        A_list.append(math.atan2(math.sin(tw - tp), math.cos(tw - tp)))
                        B_list.append(math.atan2(math.sin(tw + tp), math.cos(tw + tp)))

        # her karede yeni özellik seti (basit ve dayanıklı)
        p = cv2.goodFeaturesToTrack(gray, maxCorners=800, qualityLevel=0.01,
                                    minDistance=10)
        px = p.reshape(-1, 1, 2).astype(np.float32) if p is not None else None
        prev = gray

    cap.release()
    if len(A_list) < 30:
        print(f"YETERSİZ ÖRNEK ({len(A_list)}) — hareketli kare az")
        return

    A = np.array(A_list)
    B = np.array(B_list)
    yayA, ortA = dairesel_yayilim(A)
    yayB, ortB = dairesel_yayilim(B)

    print(f"Örnek sayısı: {len(A)}")
    print(f"ROTASYON hipotezi  (θw−θp): yayılım={yayA:.3f}  ortalama={ortA:+7.1f}°")
    print(f"YANSIMA  hipotezi  (θw+θp): yayılım={yayB:.3f}  ortalama={ortB:+7.1f}°")
    if yayA < yayB:
        print(f"→ SONUÇ: ROTASYON (normal geometri, güç oranı {yayB/max(yayA,1e-9):.1f}x)")
    else:
        print(f"→ SONUÇ: YANSIMA — eksenler AYNALI! (güç oranı {yayA/max(yayB,1e-9):.1f}x)")


if __name__ == "__main__":
    main()
