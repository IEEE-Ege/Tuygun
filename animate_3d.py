import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import mpl_toolkits.mplot3d.axes3d as p3

def main():
    print("Veriler yükleniyor...")
    df = pd.read_csv("trajectory_data.csv")
    
    # Animasyonu hızlandırmak ve dosya boyutunu küçültmek için subsample (her 10 karede bir)
    subsample = 10
    df = df.iloc[::subsample, :].reset_index(drop=True)
    
    n_frames = len(df)
    
    gt_x = df['gt_x'].values
    gt_y = df['gt_y'].values
    gt_z = df['gt_z'].values
    
    pred_x = df['pred_x'].values
    pred_y = df['pred_y'].values
    pred_z = df['pred_z'].values
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Eksen sınırlarını ayarla (tüm rotayı kapsayacak şekilde)
    ax.set_xlim(min(np.min(gt_x), np.min(pred_x)) - 10, max(np.max(gt_x), np.max(pred_x)) + 10)
    ax.set_ylim(min(np.min(gt_y), np.min(pred_y)) - 10, max(np.max(gt_y), np.max(pred_y)) + 10)
    ax.set_zlim(min(np.min(gt_z), np.min(pred_z)) - 5, max(np.max(gt_z), np.max(pred_z)) + 5)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Tuygun Odometri: 3D Animasyon')
    
    # Tüm rotayı soluk çiz (arkaplan)
    ax.plot(gt_x, gt_y, gt_z, color='blue', alpha=0.2, linestyle=':')
    ax.plot(pred_x, pred_y, pred_z, color='red', alpha=0.2, linestyle=':')
    
    # Hareket eden objeler (dronlar)
    gt_point, = ax.plot([], [], [], 'bo', markersize=8, label='Gerçek Rota (GT)')
    pred_point, = ax.plot([], [], [], 'ro', markersize=8, label='VO (Tahmin)')
    
    # Kuyruklar (son 30 noktayı çiz)
    tail_length = 30
    gt_tail, = ax.plot([], [], [], 'b-', linewidth=2, alpha=0.8)
    pred_tail, = ax.plot([], [], [], 'r-', linewidth=2, alpha=0.8)
    
    # Metin bilgisi (hata ve yükseklik)
    info_text = ax.text2D(0.05, 0.95, "", transform=ax.transAxes, fontsize=10, 
                          verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.legend(loc='upper right')
    
    def update(num):
        if num % 50 == 0:
            print(f"Frame {num}/{n_frames} render ediliyor...")
            
        start_idx = max(0, num - tail_length)
        
        # Objelerin anlık konumu
        gt_point.set_data(np.array([gt_x[num]]), np.array([gt_y[num]]))
        gt_point.set_3d_properties(np.array([gt_z[num]]))
        
        pred_point.set_data(np.array([pred_x[num]]), np.array([pred_y[num]]))
        pred_point.set_3d_properties(np.array([pred_z[num]]))
        
        # Kuyrukları güncelle
        gt_tail.set_data(gt_x[start_idx:num+1], gt_y[start_idx:num+1])
        gt_tail.set_3d_properties(gt_z[start_idx:num+1])
        
        pred_tail.set_data(pred_x[start_idx:num+1], pred_y[start_idx:num+1])
        pred_tail.set_3d_properties(pred_z[start_idx:num+1])
        
        # Hatayı hesapla
        error_3d = np.sqrt((gt_x[num]-pred_x[num])**2 + (gt_y[num]-pred_y[num])**2 + (gt_z[num]-pred_z[num])**2)
        
        # Metni güncelle
        # Orijinal frame_id (subsample * num)
        text = (f"Frame: {num * subsample}\n"
                f"3D Hata: {error_3d:.2f} m\n"
                f"GT Z: {gt_z[num]:.2f} m\n"
                f"VO Z: {pred_z[num]:.2f} m")
        info_text.set_text(text)
        
        # Kamerayı objelere doğru döndür/bak (dinamik kamera efekti)
        # Sadece azimuth'u döndürerek yavaşça etraflarında dönmesini sağlayalım
        ax.view_init(elev=20., azim=num * (360 / n_frames))
        
        return gt_point, pred_point, gt_tail, pred_tail, info_text

    print("Animasyon oluşturuluyor (biraz zaman alabilir)...")
    ani = FuncAnimation(fig, update, frames=n_frames, interval=33, blit=False)
    
    writer = FFMpegWriter(fps=30, metadata=dict(artist='Tuygun'), bitrate=1800)
    ani.save("3d_animation.mp4", writer=writer)
    print("Animasyon 3d_animation.mp4 olarak kaydedildi!")

if __name__ == "__main__":
    main()
