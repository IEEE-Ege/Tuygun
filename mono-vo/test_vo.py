import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mono_vo import MonocularVO

def main():
    print("Test başlatılıyor...")
    
    # 1. Kalibrasyon Parametreleri (1080p)
    fx = 1389.7
    fy = 1387.1
    cx = 954.007
    cy = 558.896
    
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1]
    ], dtype=np.float64)
    
    vo = MonocularVO()
    vo.K = K
    vo.update_calibration(focal=fx, pp=(cx, cy))
    
    print("CSV verisi yükleniyor...")
    df = pd.read_csv('THYZ_2026_Ornek_Veri_1_translation.csv')
    
    cap = cv2.VideoCapture('THYZ_2026_Ornek_Veri_1.MP4')
    if not cap.isOpened():
        print("Hata: Video dosyası açılamadı!")
        return

    # Yarışma Simülasyonu Ayarı
    GPS_KESILME_KARESI = 300  # Bu kareden itibaren GPS verisi 0'lanacak
    
    # Karşılaştırma için listeler
    gt_x, gt_y = [], []
    pred_x, pred_y = [], []

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Video tamamlandı veya okunamadı.")
            break
            
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if frame_idx < len(df):
            row = df.iloc[frame_idx]
            true_x = float(row['translation_x'])
            true_y = float(row['translation_y'])
            true_z = float(row['translation_z'])
            
            # 4. JSON Formatındaki API Verisini Simüle Et
            if frame_idx >= GPS_KESILME_KARESI:
                # GPS KOPTU
                json_data = {
                    "translation_x": 0.0,
                    "translation_y": 0.0,
                    "translation_z": 0.0,
                    "gps_health_status": 0
                }
                durum = "KESILDI"
            else:
                # GPS SAĞLIKLI
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
            
        # 5. Sistemi Çalıştır ve Çıktıyı Al
        sonuc = vo.process_frame(gray_frame, json_data)
        
        px = sonuc["detected_translations"][0]["translation_x"]
        py = sonuc["detected_translations"][0]["translation_y"]
        
        # Karşılaştırma için kaydet
        gt_x.append(true_x)
        gt_y.append(true_y)
        pred_x.append(px)
        pred_y.append(py)
        
        if frame_idx % 50 == 0:
            print(f"Kare {frame_idx:04d} | GPS: {durum} | GT: ({true_x:.2f}, {true_y:.2f}) -> VO: ({px:.2f}, {py:.2f})")
            
        if vo.px_ref is not None and len(vo.px_ref) > 0:
            for pt in vo.px_ref:
                cv2.circle(frame, (int(pt[0]), int(pt[1])), 3, (0, 255, 0), -1)
                
        # Ekranda durumu yaz
        renk = (0, 255, 0) if durum == "SAGLIKLI" else (0, 0, 255)
        cv2.putText(frame, f"Kare: {frame_idx} | GPS: {durum}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, renk, 2)
        cv2.putText(frame, f"Hata(X): {abs(true_x - px):.2f}m", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Hata(Y): {abs(true_y - py):.2f}m", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
        frame_resized = cv2.resize(frame, (960, 540))
        cv2.imshow("Monocular VO Test", frame_resized)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Kullanıcı tarafından durduruldu.")
            break
            
        frame_idx += 1
        
        # if frame_idx > 900:
        #     print("Hata ayıklama için 900. karede durduruluyor.")
        #     break

    cap.release()
    cv2.destroyAllWindows()
    
    print("Grafik çiziliyor...")
    plt.figure(figsize=(10, 8))
    plt.plot(gt_x, gt_y, label='Gerçek Rota (Ground Truth)', color='blue', linewidth=2)
    plt.plot(pred_x, pred_y, label='VO Tahmini', color='red', linestyle='--', linewidth=2)
    
    if len(gt_x) > GPS_KESILME_KARESI:
        plt.scatter(gt_x[GPS_KESILME_KARESI], gt_y[GPS_KESILME_KARESI], color='green', s=150, label='GPS Kesintisi (Kare 300)', zorder=5)
        
    plt.title('İHA Yörünge Karşılaştırması (Gerçek vs VO)')
    plt.xlabel('X Ekseni (m)')
    plt.ylabel('Y Ekseni (m)')
    plt.legend()
    plt.grid(True)
    plt.savefig('trajectory_comparison.png')
    print("Test tamamlandı. Karşılaştırma grafiği 'trajectory_comparison.png' olarak kaydedildi.")

if __name__ == "__main__":
    main()
