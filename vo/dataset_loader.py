import os
import pandas as pd


def load_dataset(csv_path, data_dir):
    # CSV ayracını otomatik algıla (noktalı virgül veya virgül)
    with open(csv_path, "r") as f:
        first_line = f.readline()
    sep = ";" if ";" in first_line else ","
    df = pd.read_csv(csv_path, sep=sep)

    # Eksik kolonlar için varsayılan değer ekle
    if "translation_z" not in df.columns:
        df["translation_z"] = 0.0
    # gps_health_status yoksa gps_health_for_row GPS_KESILME_KARESI'ni kullanır

    # Video dosyası kontrolü
    is_video = False
    if os.path.isfile(data_dir) and data_dir.lower().endswith(('.mp4', '.avi', '.mov')):
        is_video = True
        print(f"Video dosyası algılandı: {data_dir}")
        return df, is_video, data_dir

    # Görüntü uzantısını otomatik algıla (.webp / .jpg / .png)
    sample = str(df.iloc[0]["frame_numbers"])
    img_ext = ".webp"
    for ext in [".webp", ".jpg", ".jpeg", ".png"]:
        if os.path.exists(os.path.join(data_dir, sample + ext)):
            img_ext = ext
            break
    print(f"Görüntü uzantısı: {img_ext}")

    df["frame_path"] = df["frame_numbers"].apply(
        lambda fn: os.path.join(data_dir, str(fn) + img_ext)
    )
    mask = df["frame_path"].apply(os.path.exists)
    missing = (~mask).sum()
    if missing:
        print(f"UYARI: {missing} kare dosyası bulunamadı, atlanıyor.")
        df = df[mask].reset_index(drop=True)
    return df, is_video, data_dir


def gps_health_for_row(row, frame_idx, has_health_col, gps_kesilme_karesi):
    """
    Satırdan GPS sağlık durumu döner.
    CSV'de gps_health_status varsa doğrudan okur.
    Yoksa GPS_KESILME_KARESI simülasyonunu kullanır.
    """
    if has_health_col:
        return int(row.get("gps_health_status", 1))
    if gps_kesilme_karesi < 0:
        return 1
    return 0 if frame_idx >= gps_kesilme_karesi else 1
