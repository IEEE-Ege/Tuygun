import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mono_vo import MonocularVO
import time

def main():
    print("Hizli test baslatiliyor (imshow kapalı)...")
    
    fx, fy, cx, cy = 1389.7, 1387.1, 954.007, 558.896
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    
    vo = MonocularVO()
    vo.K = K
    vo.update_calibration(focal=fx, pp=(cx, cy))
    
    print("CSV verisi yukleniyor...")
    df = pd.read_csv('THYZ_2026_Ornek_Veri_1_translation.csv')
    
    cap = cv2.VideoCapture('THYZ_2026_Ornek_Veri_1.MP4')
    if not cap.isOpened():
        print("Hata: Video dosyasi acilamadi!")
        return

    GPS_KESILME_KARESI = 300
    
    gt_x, gt_y = [], []
    pred_x, pred_y = [], []

    frame_idx = 0
    start_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Video tamamlandi.")
            break
            
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if frame_idx < len(df):
            row = df.iloc[frame_idx]
            true_x = float(row['translation_x'])
            true_y = float(row['translation_y'])
            true_z = float(row['translation_z'])
            
            if frame_idx >= GPS_KESILME_KARESI:
                json_data = {
                    "translation_x": 0.0,
                    "translation_y": 0.0,
                    "translation_z": 0.0,
                    "gps_health_status": 0
                }
                durum = "KESILDI"
            else:
                json_data = {
                    "translation_x": true_x,
                    "translation_y": true_y,
                    "translation_z": true_z,
                    "gps_health_status": 1
                }
                durum = "SAGLIKLI"
        else:
            print("CSV verisi bitti.")
            break
            
        sonuc = vo.process_frame(gray_frame, json_data)
        
        px = sonuc["detected_translations"][0]["translation_x"]
        py = sonuc["detected_translations"][0]["translation_y"]
        
        gt_x.append(true_x)
        gt_y.append(true_y)
        pred_x.append(px)
        pred_y.append(py)
        
        if frame_idx % 200 == 0:
            elapsed = time.time() - start_time
            fps = (frame_idx + 1) / elapsed if elapsed > 0 else 0
            print(f"Kare {frame_idx:05d} | GPS: {durum} | GT: ({true_x:.2f}, {true_y:.2f}) -> VO: ({px:.2f}, {py:.2f}) | {fps:.1f} FPS")
            
        frame_idx += 1

    cap.release()
    
    elapsed = time.time() - start_time
    print(f"\nToplam {frame_idx} kare, {elapsed:.1f} saniye ({frame_idx/elapsed:.1f} FPS)")
    
    print("Grafik ciziliyor...")
    plt.figure(figsize=(10, 8))
    plt.plot(gt_x, gt_y, label='Gercek Rota (Ground Truth)', color='blue', linewidth=2)
    plt.plot(pred_x, pred_y, label='VO Tahmini', color='red', linestyle='--', linewidth=2)
    
    if len(gt_x) > GPS_KESILME_KARESI:
        plt.scatter(gt_x[GPS_KESILME_KARESI], gt_y[GPS_KESILME_KARESI], color='green', s=150, label='GPS Kesintisi (Kare 300)', zorder=5)
        
    plt.title('IHA Yorunge Karsilastirmasi (Gercek vs VO)')
    plt.xlabel('X Ekseni (m)')
    plt.ylabel('Y Ekseni (m)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.savefig('trajectory_comparison.png')
    print("Test tamamlandi. Grafik 'trajectory_comparison.png' olarak kaydedildi.")

if __name__ == "__main__":
    main()
