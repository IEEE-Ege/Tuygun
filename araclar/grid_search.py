import subprocess
import re
import os

best_rmse = float('inf')
best_scale = 0

for scale in [0.45, 0.50, 0.55, 0.65, 0.70]:
    env = {"PPM_X": str(scale), "PPM_Y": str(scale)}
    result = subprocess.run(["python3", "araclar/fast_test.py"], env={**os.environ, **env}, capture_output=True, text=True)
    out = result.stdout
    match = re.search(r"Toplam 3D RMSE\s*:\s*([\d.]+)", out)
    if match:
        rmse = float(match.group(1))
        print(f"Scale={scale} -> RMSE={rmse}")
        if rmse < best_rmse:
            best_rmse = rmse
            best_scale = scale
    else:
        print(f"Scale={scale} -> FAILED")

print(f"BEST: Scale={best_scale} -> RMSE={best_rmse}")
