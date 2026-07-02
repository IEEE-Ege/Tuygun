#!/bin/zsh
# Çift tıkla: kareler_termal/ klasöründeki hazır kareleri işler, sonucu gösterir.
cd "$(dirname "$0")/.."
python3 frame_sistemi/termal_framebyframe.py
echo ""
echo "Bitti — kapatmak için bu pencereyi kapatabilirsiniz."
