import pandas as pd
import numpy as np

df = pd.read_csv('THYZ_2026_Ornek_Veri_1_translation.csv')

# Ardisik kare farklari
df['dist'] = np.sqrt(df['translation_x'].diff()**2 + df['translation_y'].diff()**2 + df['translation_z'].diff()**2)

print("GPS saglikliyken (ilk 300 kare) kare-kare mesafe istatistikleri:")
gps_ok = df.iloc[1:300]
print(f"  Ortalama: {gps_ok['dist'].mean():.6f} m/kare")
print(f"  Maksimum: {gps_ok['dist'].max():.6f} m/kare")
print(f"  Toplam: {gps_ok['dist'].sum():.4f} m")

print()
print("GPS kesildikten sonra (300-1000) kare-kare mesafe istatistikleri:")
gps_off = df.iloc[300:1000]
print(f"  Ortalama: {gps_off['dist'].mean():.6f} m/kare")
print(f"  Maksimum: {gps_off['dist'].max():.6f} m/kare")

print()
print("1000-2000 arasi:")
chunk = df.iloc[1000:2000]
print(f"  Ortalama: {chunk['dist'].mean():.6f} m/kare")
print(f"  Kum. Y: {df.iloc[1000]['translation_y']:.2f} -> {df.iloc[2000]['translation_y']:.2f}")

print()
print("Drone hizi belirli karelerde:")
for frame in [100, 200, 300, 500, 700, 800, 900, 1000, 1500, 2000, 3000, 5000, 8000, 10000]:
    if frame < len(df) and frame > 0:
        d = df.iloc[frame]['dist']
        y = df.iloc[frame]['translation_y']
        print(f"  Kare {frame:5d}: hiz={d:.6f} m/kare, y={y:.2f} m")
