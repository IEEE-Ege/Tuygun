#!/bin/zsh
# Çift tıkla: RGB videoyu işler, metrikleri basar, grafiği çizer ve açar.
cd "$(dirname "$0")/.."
python3 video_sistemi/video_main.py --sensor rgb
echo ""
echo "Bitti — kapatmak için bu pencereyi kapatabilirsiniz."
