#!/bin/zsh
# Çift tıkla: Termal videoyu işler, metrikleri basar, grafiği çizer ve açar.
cd "$(dirname "$0")/.."
python3 video_sistemi/video_main.py --sensor thermal
echo ""
echo "Bitti — kapatmak için bu pencereyi kapatabilirsiniz."
