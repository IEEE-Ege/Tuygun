"""RGB kare klasörünü frame-by-frame işler.

Kullanım:  python3 frame_sistemi/rgb_framebyframe.py
Ayarlar :  frame_sistemi/frame_ayarlar.py → RGB bölümü
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frame_sistemi.frame_pipeline import calistir
from frame_sistemi.frame_ayarlar import RGB

if __name__ == "__main__":
    calistir("rgb", RGB)
