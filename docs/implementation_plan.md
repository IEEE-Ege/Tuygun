# Odometri Optimizasyon Planı

Kullanıcının gönderdiği grafikler üzerinde yapılan analizler sonucunda üç temel problem tespit edilmiştir ve aşağıdaki düzeltmeler uygulanacaktır:

## 1. Yaw (Dönüş) Asimetrisi ve Heading Sürüklenmesi
**Problem:** 2D yörüngenin sonlarına doğru (kare 7000-9000 arası) VO'nun (kırmızı) gerçek rotaya (mavi) göre daha "düz" ve geniş bir kavis çizdiği görülüyor. Kodu incelediğimde, her karede uygulanan `0.02` derecelik bir *deadband (ölü bölge)* var. Üstelik bu deadband değeri eşiği aşan dönüşlerden de çıkartılıyor (`yaw - deadband`). Bu durum, dronun yaptığı tüm yavaş dönüşlerin sistematik olarak eksik hesaplanmasına ve rotanın dışa doğru savrulmasına (under-estimation) neden olmaktadır.
**Çözüm:** 
- Deadband mantığını kaldıracağız veya sadece çok küçük gürültüler için bırakıp ana `yaw` değerinden çıkartılmasını engelleyeceğiz. Dönüşlerin eksiksiz entegre edilmesi heading (yön) sapmasını büyük oranda çözecektir.

## 2. Güvenilirlik (Confidence) Skorunun Çökmesi
**Problem:** `Hata & Güvenilirlik Zaman Serisi` grafiğinde yeşil renkli Confidence çizgisi 3000. kareden sonra sıfıra iniyor ve bir daha çıkmıyor. Oysa log kayıtlarında Optical Flow'un çok sağlıklı çalıştığı (`inlier_ratio=1.00`) görülüyor. Sistemin sırf GPS olmadan uzun süre uçtuğu (`dr_frame_count`) için güvenilirliği sıfıra indirmesi, grafikteki analizleri yanıltıyor.
**Çözüm:** 
- `_compute_confidence` içindeki zamana bağlı (`dr_frame_count`) aşırı brutal cezalandırma (decay) faktörü yumuşatılacak (örn. minimum %40'a kadar düşmesine izin verilecek, sıfırlanmayacak). Böylece optik akış iyi çalıştığı sürece güvenilirlik dip yapmayacak.

## 3. Scale Drift (Ölçek) Çarpanı Revizyonu
**Problem:** `smoothed_scale = 1.0 + (scale - 1.0) * 1.3` formülündeki `1.3` katsayısı bir filtre (EMA) olmak yerine değişimi %30 güçlendiren bir amplifikatör görevi görüyor. X ve Y eksenlerindeki ilk büyük çukurda VO'nun -200'e kadar abartılı inmesinin (over-estimation) sebebi bu abartılmış ölçek çarpanı olabilir.
**Çözüm:** 
- Katsayıyı `1.3` yerine `0.8` gibi gerçek bir yumuşatma filtresine (low-pass) çevirerek ani sıçramaların ve abartılı ölçek büyümelerinin önüne geçilecek.

Lütfen bu planı onaylayın, ardından kod üzerinde değişiklikleri uygulayıp testleri çalıştıracağım.
