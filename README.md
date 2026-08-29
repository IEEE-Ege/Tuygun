# Tuygun Odometri Projesi (MonoVO)

Bu proje, tek kameradan alınan (monoküler) görüntüler ve GPS kesintisi durumlarında kullanılmak üzere tasarlanmış bir **Görsel Odometri (Visual Odometry - VO)** çözümüdür. Drone veya benzeri hava araçları için, GPS'in geçici olarak koptuğu durumlarda (Dead-Reckoning) sistemin kendi konumunu tahmin etmesini sağlar.

## 🏗️ Mimari Özeti

Sistem temel olarak **FAST özellik tespiti (Feature Detection)** ve **Lucas-Kanade optik akış (Optical Flow)** yöntemlerini kullanarak ardışık kareler (frame) arasındaki piksel yer değiştirmesini hesaplar.

1. **GPS Fazı (Kalibrasyon):**
   - Sistem başlatıldığında veya GPS sinyali sağlıklıyken, GPS'den gelen gerçek dünya hareketleri (metre cinsinden `dEasting`, `dNorthing`) ile kameradan hesaplanan piksel hareketleri (pixel displacement) karşılaştırılır.
   - Bu karşılaştırma ile **Piksel-Metre Çarpanı (PPM - Pixel Per Meter)** hesaplanır. Bu çarpan, 1 pikselin gerçek dünyada kaç metreye denk geldiğini ifade eder. X ve Y eksenleri için bağımsız olarak hesaplanıp EMA (Üstel Hareketli Ortalama) filtresi ile yumuşatılır.
   - Aynı zamanda, drone'un **Heading (yönelimi)** de GPS hareket vektörü ile optik akış vektörü karşılaştırılarak kalibre edilir.

2. **Dead-Reckoning Fazı (GPS Kesintisi):**
   - GPS sinyali kesildiğinde, sistem tamamen optik akışa güvenir.
   - Ardışık karelerde özellikler eşleştirilir. Gürültülü eşleşmeleri (outliers) ayıklamak için **RANSAC ile 2D Afin Dönüşüm** (`estimateAffinePartial2D`) kullanılır.
   - Hesaplanan `dx` ve `dy` (piksel kaymaları) mevcut `PPM_X` ve `PPM_Y` değerleri ile metreye çevrilir.
   - İki ardışık kare arasındaki affine ölçek değişimi (`scale`), yükseklik (Z) değişimini hesaplamak için kullanılır.

## ⚙️ Teknik Detaylar ve Karşılaşılan Zorluklar

- **X Ekseni Sapmaları (Yanal Hata):** Drone'un ileri yönlü uçuş karakteristiği nedeniyle optik akış genelde Y ekseninde (aşağı-yukarı) baskındır. Yanal eksende (X) oluşan ufak açılı salınımlar (roll), görüntüde çeviri (translation) gibi yorumlanıp `VO_X`'in gerçek değerden fazla kaymasına neden olabilir. Bu durumu minimize etmek için `smoothed_scale` (ölçek EMA çarpanı) parametresi optimize edilmiştir.
- **Dairesel Yön (Heading) Güncellemesi:** Afin dönüşümden elde edilen rotasyon, kameranın (veya drone'un) Z eksenindeki yalpalamasını (roll) değil, görüntü düzlemindeki dönüşü ifade eder. Drone'un gerçek Heading (Yönelimi) bilgisi, optik akış gürültüsünden kolayca etkilenebildiği için ölü bant (deadband) yöntemi ile filtrelenir.
- **Güvenilirlik Skoru (Confidence):** VO algoritması sürekli olarak inlier oranı, özellik sayısı ve ardışık karelerdeki hız gibi metrikleri izleyerek bir `Confidence` değeri (0.0 - 1.0) üretir. Eşik değerin altına düştüğünde sistem uyarılır (Unreliable).

## 🚀 Kullanım

Test senaryosunu çalıştırmak için:
```bash
python3 fast_test.py
```
Bu betik veri kümesini okur, yörüngeyi tahmin eder, metrikleri (3D RMSE) hesaplar ve sonuçları 2D/3D grafikler içeren `trajectory_comparison_2026_v1.png` dosyasına çizer.

## 📌 Son Güncellemeler (Hafıza Dosyası)

- PPM (Pixel-per-meter) hesabı `ppm_x` ve `ppm_y` olarak ayrıldı. Bu sayede X ve Y eksenlerinin farklı karakteristikte olması problemi çözülmeye çalışıldı.
- `fast_test.py` grafiğine **3 Boyutlu Yörünge Karşılaştırması** eklendi.
- `mono_vo.py` içindeki `smoothed_scale` çarpanı `1.3` katsayısına çekilerek 3D RMSE'nin 29 metre civarında tutulması sağlandı. (Bu parametre özellikle X eksenindeki "overshoot" hatalarını kompanze etmektedir).
- `fast_test.py` içindeki GridSpec yapısı 3D grafiği barındıracak şekilde düzenlendi.
