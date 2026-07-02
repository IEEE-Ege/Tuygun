#!/bin/zsh
# Çift tıkla: kareler_rgb/ klasöründeki hazır kareleri işler, sonucu gösterir.
cd "$(dirname "$0")/.."
python3 frame_sistemi/rgb_framebyframe.py
echo ""
echo "Bitti — kapatmak için bu pencereyi kapatabilirsiniz."
