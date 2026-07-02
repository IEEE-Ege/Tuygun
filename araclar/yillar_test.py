# ═══════════════════════════════════════════════════════════════════════════
#  YILLAR TESTİ — yillar/ altındaki TÜM yıl klasörlerini otomatik tarar,
#  video+CSV çiftlerini eşleştirir ve her çifti AYRI bir Terminal
#  penceresinde PARALEL koşar. Her koşu bitince kendi grafiğini açar.
#
#      python3 araclar/yillar_test.py             # hepsini paralel başlat
#      python3 araclar/yillar_test.py --liste     # sadece bulunan çiftleri göster
#      python3 araclar/yillar_test.py --sirali    # tek terminalde sırayla koş
#      python3 araclar/yillar_test.py --kesilme 1500   # GPS kesinti karesi (hepsine)
#      python3 araclar/yillar_test.py --yil 2025 # sadece bir yılı koş
#
#  Keşif kuralları:
#    • Video: yillar/<yıl>/ altında (özyinelemeli) .mp4/.avi/.mov
#    • CSV eşleşmesi (aynı klasörde, öncelik sırasıyla):
#        <video>-translation.csv → <video>_translation.csv → <video>.csv
#        → klasördeki TEK csv (2024'teki GT_Translations.csv durumu)
#    • Sensör: dosya adında 'termal'/'thermal' geçiyorsa termal, yoksa RGB
#    • Kalibrasyon: video klasöründen yıl klasörüne doğru aranan
#      *alibrasyon*.txt (termal video için adında termal geçen tercih edilir)
#
#  Çıktı adlandırma: her koşu, zaman damgalı KENDİ klasörüne yazar —
#  üst üste yazma hiçbir koşulda mümkün değil:
#      video_sistemi/cikti/yillar_YYYYAAGG_SSDDss/<yıl>_<video-adı>.png
# ═══════════════════════════════════════════════════════════════════════════
import argparse
import os
import re
import shlex
import subprocess
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_ROOT)

YILLAR_DIR = "yillar"
VIDEO_UZANTILAR = (".mp4", ".avi", ".mov")


def csv_bul(video_path):
    """Videonun yanındaki ground-truth CSV'yi bulur (öncelik sıralı)."""
    klasor = os.path.dirname(video_path)
    govde = os.path.splitext(os.path.basename(video_path))[0]
    adaylar = [
        os.path.join(klasor, govde + "-translation.csv"),
        os.path.join(klasor, govde + "_translation.csv"),
        os.path.join(klasor, govde + ".csv"),
    ]
    for aday in adaylar:
        if os.path.exists(aday):
            return aday
    tum_csv = [os.path.join(klasor, f) for f in sorted(os.listdir(klasor))
               if f.lower().endswith(".csv")]
    if len(tum_csv) == 1:
        return tum_csv[0]
    return None  # birden çok csv varsa tahmin etme — belirsizliği raporla


def kalibrasyon_bul(video_path, yil_dir, termal):
    """Video klasöründen yıl klasörüne doğru kalibrasyon txt arar."""
    klasor = os.path.dirname(video_path)
    yil_dir = os.path.abspath(yil_dir)
    while True:
        txtler = [os.path.join(klasor, f) for f in sorted(os.listdir(klasor))
                  if "alibrasyon" in f.lower() and f.lower().endswith(".txt")]
        if txtler:
            termal_olan = [t for t in txtler if re.search(r"termal|thermal", t, re.I)]
            digerleri = [t for t in txtler if t not in termal_olan]
            if termal and termal_olan:
                return termal_olan[0]
            if not termal and digerleri:
                return digerleri[0]
            return txtler[0]
        if os.path.abspath(klasor) == yil_dir:
            return None
        klasor = os.path.dirname(klasor)


def ciftleri_kesfet(yil_filtresi=None):
    ciftler, sorunlar = [], []
    if not os.path.isdir(YILLAR_DIR):
        print(f"HATA: '{YILLAR_DIR}/' klasörü yok.")
        sys.exit(1)
    for yil in sorted(os.listdir(YILLAR_DIR)):
        yil_dir = os.path.join(YILLAR_DIR, yil)
        if not os.path.isdir(yil_dir) or yil.startswith("."):
            continue
        if yil_filtresi and yil != yil_filtresi:
            continue
        for kok, _dirs, dosyalar in os.walk(yil_dir):
            for dosya in sorted(dosyalar):
                if not dosya.lower().endswith(VIDEO_UZANTILAR):
                    continue
                video = os.path.join(kok, dosya)
                termal = bool(re.search(r"termal|thermal", dosya, re.I))
                csv = csv_bul(video)
                kalib = kalibrasyon_bul(video, yil_dir, termal)
                if csv is None:
                    sorunlar.append(f"{video}: eşleşen CSV bulunamadı (ya da birden çok aday var)")
                    continue
                if kalib is None:
                    sorunlar.append(f"{video}: kalibrasyon txt bulunamadı")
                    continue
                ciftler.append(dict(yil=yil, video=video, csv=csv,
                                    kalibrasyon=kalib, termal=termal))
    return ciftler, sorunlar


def komut_uret(cift, cikti_dir, kesilme):
    govde = os.path.splitext(os.path.basename(cift["video"]))[0]
    etiket = f"{cift['yil']}_{govde}"
    out_fig = os.path.join(cikti_dir, etiket + ".png")
    # Genel model: yıla özel değil — termal→nötr termal profili, RGB→rgb.
    # Kalibrasyon her koşuda videonun kendi yılının txt'sinden okunur.
    sensor = "thermal_2025" if cift["termal"] else "rgb"
    parcalar = [
        sys.executable, "video_sistemi/video_main.py",
        "--sensor", sensor,
        "--video", cift["video"],
        "--csv", cift["csv"],
        "--kalibrasyon", cift["kalibrasyon"],
        "--cikti", out_fig,
    ]
    if kesilme is not None:
        parcalar += ["--kesilme", str(kesilme)]
    return etiket, parcalar


def terminalde_ac(etiket, parcalar):
    """Komutu yeni bir macOS Terminal penceresinde başlatır."""
    ic_komut = "cd " + shlex.quote(_ROOT) + " && " + " ".join(shlex.quote(p) for p in parcalar)
    # Pencere başlığında etiket görünsün, bitince pencere açık kalsın
    tam = f"echo '═══ {etiket} ═══' && {ic_komut}"
    osa = f'tell application "Terminal" to do script "{tam}"'.replace("\\", "\\\\")
    subprocess.run(["osascript", "-e", 'tell application "Terminal" to activate',
                    "-e", osa], check=True)


def main():
    ap = argparse.ArgumentParser(description="yillar/ altındaki tüm video+CSV çiftlerini paralel test eder")
    ap.add_argument("--liste", action="store_true", help="Koşma, sadece bulunan çiftleri göster")
    ap.add_argument("--sirali", action="store_true",
                    help="Yeni pencere açmadan bu terminalde sırayla koş")
    ap.add_argument("--kesilme", type=int, default=None, help="GPS kesinti karesi (tüm koşulara)")
    ap.add_argument("--yil", default=None, help="Sadece bu yılı koş (örn. 2025)")
    args = ap.parse_args()

    ciftler, sorunlar = ciftleri_kesfet(args.yil)

    print(f"{len(ciftler)} video+CSV çifti bulundu:\n")
    for c in ciftler:
        tip = "TERMAL" if c["termal"] else "RGB   "
        print(f"  [{c['yil']}] {tip}  {os.path.basename(c['video'])}")
        print(f"          csv:   {os.path.basename(c['csv'])}")
        print(f"          kalib: {os.path.basename(c['kalibrasyon'])}")
    for s in sorunlar:
        print(f"  UYARI: {s}")
    if args.liste or not ciftler:
        return

    cikti_dir = os.path.join("video_sistemi", "cikti",
                             time.strftime("yillar_%Y%m%d_%H%M%S"))
    os.makedirs(cikti_dir, exist_ok=True)
    print(f"\nÇıktı klasörü: {cikti_dir}/  (her koşunun grafiği + yörünge CSV'si burada)")

    for cift in ciftler:
        etiket, parcalar = komut_uret(cift, cikti_dir, args.kesilme)
        if args.sirali:
            print(f"\n{'═'*70}\n  {etiket}\n{'═'*70}")
            subprocess.run(parcalar)
        else:
            print(f"  → Terminal penceresi açılıyor: {etiket}")
            terminalde_ac(etiket, parcalar)

    if not args.sirali:
        print("\nTüm testler ayrı pencerelerde koşuyor. Her biri bitince kendi"
              "\ngrafiğini otomatik açacak; metrikler pencerelerin sonunda.")


if __name__ == "__main__":
    main()
