# Tuygun Odometri - Mimari ve Sistem Hafızası (Memory)

Bu doküman, Tuygun İHA Odometri projesinin (Visual Odometry) mevcut mimarisini, algoritma yapısını, karşılaşılan problemleri ve bu problemlere üretilen çözümleri kayıt altına almak (hafıza oluşturmak) amacıyla hazırlanmıştır. Gelecekteki geliştirmelerde "neden bu şekilde yapıldığına" dair rehber niteliğindedir.

## 1. Proje Amacı
Bu sistem, dronun GPS sinyali alamadığı durumlarda üzerindeki monoküler (tek) kamera aracılığıyla elde ettiği görüntüleri kullanarak kendi 3 boyutlu (X, Y, Z) konumunu tahmin etmesini (dead-reckoning) sağlar. 

## 2. Temel Mimari ve Dosyalar

*   **`mono_vo.py`**: Sistemin kalbi olan optik akış (Optical Flow) ve görsel odometri motorudur. `MonocularVO` sınıfını içerir.
    *   **Özellik Çıkarımı (Feature Detection):** Görüntüyü grid'lere (ızgaralara) bölerek homojen dağılımlı özellik noktaları bulur (Shi-Tomasi vb.).
    *   **Özellik Takibi (Feature Tracking):** Lucas-Kanade (KLT) optik akış algoritması ile ardışık kareler (frame) arasında özellik noktalarının hareketini takip eder.
    *   **Hareket Tahmini:** RANSAC destekli kısmi Afin dönüşüm (`estimateAffinePartial2D`) kullanarak iki kare arasındaki öteleme (dx, dy), dönüş açısı ve ölçek (scale) değişimini hesaplar.
    *   **Online Kalibrasyon (GPS varken):** GPS verisi sağlıklı iken, piksel hareketini (px) gerçek dünya metre hareketine (m) çeviren **PPM (Pixel-Per-Meter)** oranını ve dronun ilerleme açısını (Heading) dinamik olarak öğrenir.
    *   **Dead-Reckoning (GPS kesildiğinde):** GPS kesildiği an öğrenilmiş olan PPM ve Heading değerlerini kullanarak piksel hareketinden X, Y ve Z eksenindeki gerçek yer değiştirmeyi hesaplamaya başlar.

*   **`fast_test.py`**: Geliştirilen algoritmanın test edilmesi, doğrulanması ve görselleştirilmesi için kullanılır.
    *   Veri setini hızlıca okur, `MonocularVO` sınıfını çalıştırır.
    *   Ground Truth (Gerçek Rota) ile tahmin edilen rotayı karşılaştırır.
    *   2D, Z-ekseni ve Toplam 3D RMSE (Hata) skorlarını hesaplar.
    *   Yörünge, Güvenilirlik (Confidence) ve eksen bazlı zaman serisi grafiklerini çizer.

## 3. Karşılaşılan Zorluklar ve Çözümler (Hafıza Notları)

### A. Scale Drift ve Survival Bias
**Problem:** Optik akış sırasında ileri doğru giderken kenardaki noktalar kaybolur (survival bias), merkeze yakın noktalar hayatta kalır. Bu durum, yanlış bir şekilde dronun küçüldüğü / uzaklaştığı (scale < 1.0) sanrısına yol açar.
**Çözüm:** Scale (ölçek) değerindeki ani sıçramaları önlemek için EMA (Exponential Moving Average) filtresi kullanıldı ve uç ölçek değerleri limitlendi (`0.90 < scale < 1.10`).

### B. Z Ekseni (İrtifa) Tahmini ve Yalancı Çukur (False Dip)
**Problem:** Sadece 2D kamera hareketi kullanılarak yükseklik (Z ekseni) tahmini yapıldığında, dronun pitch/roll hareketleri veya yerdeki düzensizlikler, dronun aslında alçalıyormuş gibi algılanmasına (scale'in büyümesine) sebep oluyordu. İlk aşamada 4000-6000 kareleri arasındaki *gerçek çukuru* yakalamak için alçalma (negatif `delta_z`) katsayısı `3.0` olarak abartıldı. Ancak bu, 450-2000 kareleri arasındaki ufak gürültüleri büyüterek devasa bir "Yalancı Çukur" (V şekli) oluşmasına neden oldu.
**Çözüm:** "Oran orantı katsayısı çok geniş" uyarısı üzerine, `3.0` olan alçalma çarpanı `1.0`'a çekildi. Tepe/yükselme çarpanı ise `0.6` olarak bırakıldı. Böylelikle hem 450-2000 arasındaki grafik "düz" (gerçeğe uygun) hale getirildi, hem de asıl çukur bölgesi doğru yönde tahmin edilmeye devam etti.

### C. Z Ekseni Hata Metriği (RMSE) Eksikliği
**Problem:** Z eksenindeki iyileştirmelerin test kodunda toplam skora etki etmediği fark edildi.
**Çözüm:** `fast_test.py` içindeki RMSE hesaplaması sadece `X` ve `Y` (2D) üzerinden yapılıyordu. Test koduna `pred_z` ve `gt_z` dahil edilerek **3D RMSE** ve bağımsız **Z RMSE** hesaplamaları eklendi.

### D. GPS / Dead-Reckoning Pürüzsüz Geçişi
**Problem:** GPS kesildiğinde koordinatların ve irtifanın (Z) aniden sıçraması.
**Çözüm:** GPS'in sağlıklı olduğu son ana kadarki `translation_x, y, z` verileri `initial_gps` ve `dr_start_z` gibi offset (ofset) değişkenleriyle tutuldu. Kesinti yaşandığı an (örneğin 450. kare), pozisyon bu kilitli offsetler üzerinden üstüne eklenerek ilerletildi (Seamless transition).

### E. 3 Boyutlu Yörünge ve Animasyon Geliştirmesi
**Problem:** Z ekseni ve 2D yörünge grafiklerinin ayrı ayrı incelenmesi, gerçek rotanın (GT) ve tahminin (VO) bütüncül davranışını kavramayı zorlaştırıyordu.
**Çözüm:** `fast_test.py` içerisine matplotlib ile 3D yörünge grafiği (X, Y, Z uzayı) entegre edildi. Ayrıca, bu rotayı zaman içinde hareket eden iki obje olarak görselleştiren `animate_3d.py` scripti eklendi. Sistem, tahmin edilen verileri `trajectory_data.csv` olarak dışa aktararak animasyonun MP4 (30 FPS) olarak render edilmesini sağladı.

## 4. Gelecek Adımlar
- **Dönüş Asimetrisi:** Hızlı dönüşlerdeki (yaw) piksel kaymalarının doğrusal harekete karışmasını önlemek için jiroskop (IMU) destekli veya daha sofistike Feature Rejection metotları incelenebilir.
- **PPM Kalibrasyonu:** Hızın sıfıra yaklaştığı anlarda (durma/hover) PPM öğrenme katsayısının askıya alınması (ağırlığının düşürülmesi).
- **Z Ekseni Düzeltmeleri:** Odometri tamamen düz zemin (Flat Earth) varsayımında hatalı ölçek verebilir, zemin yükseklik farklarını (Digital Elevation Model) hesaba katan filtreler tasarlanabilir.
