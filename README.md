# Tuygun İHA - Visual Odometry (Görsel Odometri)

Bu proje, Tuygun İnsansız Hava Aracı (İHA) için geliştirilmiş, GPS sinyali kaybolduğunda (Dead-Reckoning) dronun 3 boyutlu (X, Y, Z) konumunu sadece alt kameradan alınan görüntüler (Monoküler Optik Akış) ile tahmin etmesini sağlayan bir Görsel Odometri (Visual Odometry) sistemidir.

## Özellikler

- **Optik Akış (Optical Flow):** Lucas-Kanade (KLT) algoritması ile özellik noktalarının takibi.
- **Dinamik Ölçekleme (PPM Öğrenme):** GPS sağlıklı olduğu anlarda (ilk 450 kare) piksel hareketlerini gerçek dünya metre hareketine çeviren katsayıların dinamik olarak öğrenilmesi.
- **Kademeli Z Ekseni (İrtifa) Düzeltmesi:** Uçuş süresi (drift frame count) uzadıkça dinamik olarak artan Z ekseni telafi katsayısı.
- **Kesintisiz Geçiş (Seamless Transition):** GPS koptuğu an son güvenilir koordinattan (X, Y, Z) başlayarak rotanın sürdürülmesi.
- **3D Yörünge Animasyonu:** Uçuş rotasının ve tahminin 3 boyutlu uzayda, kamera takipli bir şekilde (Blender benzeri) MP4 formatında animasyon olarak oluşturulabilmesi.

## Dosya Yapısı

- `mono_vo.py`: Görsel odometri algoritmasını barındıran çekirdek modül.
- `fast_test.py`: Hızlı test ve doğrulama aracı. Yörüngeleri analiz eder, hataları hesaplar (RMSE) ve statik 2D/3D grafikleri çizer. Aynı zamanda animasyon için gerekli olan `trajectory_data.csv` verisini dışa aktarır.
- `animate_3d.py`: Test kodu sonucunda çıkan yörüngeyi işleyerek hareketli 3 boyutlu bir uçuş animasyonu (MP4) haline getiren araç.
- `MIMARI_VE_HAFIZA.md`: Proje süresince karşılaşılan sorunları (scale drift, Z asimetrisi, survival bias) ve bunlara yönelik uygulanan çözümleri kaydeden mimari doküman.

## Kurulum ve Kullanım

### Gereksinimler

Projenin çalışması için aşağıdaki kütüphanelere ihtiyaç vardır:
```bash
pip install numpy opencv-python matplotlib pandas
```

*Not: Animasyon oluşturmak istiyorsanız sisteminizde `ffmpeg` yüklü olmalıdır.*

### Test ve Görselleştirme

Tüm testi (9000 kare) çalıştırmak ve statik sonuç grafiğini (`trajectory_comparison.png`) ile `trajectory_data.csv` verisini oluşturmak için:

```bash
python3 fast_test.py
```

### 3 Boyutlu Animasyon Oluşturma

Öncelikle yukarıdaki `fast_test.py` adımının tamamlandığından ve `trajectory_data.csv` dosyasının dizinde olduğundan emin olun. Ardından:

```bash
python3 animate_3d.py
```

Bu kod 3 boyutlu bir `3d_animation.mp4` video dosyası oluşturacaktır.

## İletişim & Katkı
Bu repo IEEE-Ege bünyesindeki Tuygun takımı için oluşturulmuştur. Daha fazla bilgi için takımla iletişime geçebilirsiniz.
