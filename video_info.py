"""
Adım 1: Videoyu analiz et
Kullanım: python video_info.py video_dosyan.mp4
"""
import cv2
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "video.mp4"
cap  = cv2.VideoCapture(path)

fps    = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
dur    = total / fps if fps > 0 else 0

# İlk kareyi oku, gri seviye doku zenginliğini ölç
ret, frame = cap.read()
if ret:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    # Kamera profil tahmini
    if width == 640 and height == 512:
        profile = "thermal"
    elif width >= 3000:
        profile = "rgb_4k"
    else:
        profile = "rgb_1080p"
else:
    lap_var = 0
    profile = "bilinmiyor"

cap.release()

print("=" * 45)
print(f"  Dosya          : {path}")
print(f"  Çözünürlük     : {width}x{height}")
print(f"  FPS            : {fps:.2f}")
print(f"  Toplam kare    : {total}")
print(f"  Süre           : {dur:.1f} saniye  ({dur/60:.1f} dk)")
print(f"  Kamera profili : {profile}")
print(f"  Doku zenginliği: {lap_var:.1f}  (>200 iyi, <50 zayıf)")
print("=" * 45)
print("\nBu çıktıyı Claude'a yapıştır.")
