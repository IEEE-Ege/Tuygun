import re

with open('/Users/serhanensar/Desktop/Tuygun-odometri/fast_test.py', 'r') as f:
    content = f.read()

csv_code = """
    # ── CSV Kaydetme ─────────────────────────────────────────────────────────
    print("Yörünge verileri trajectory_data.csv olarak kaydediliyor...")
    import csv
    with open("trajectory_data.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "gt_x", "gt_y", "gt_z", "pred_x", "pred_y", "pred_z", "conf"])
        for i in range(len(gt_x)):
            writer.writerow([i, gt_x[i], gt_y[i], gt_z[i], pred_x[i], pred_y[i], pred_z[i], conf_list[i]])
"""

# Insert before plotting
content = content.replace('    # ── Grafik ──────────────────────────────────────────────────────────────', csv_code + '\n    # ── Grafik ──────────────────────────────────────────────────────────────')

with open('/Users/serhanensar/Desktop/Tuygun-odometri/fast_test.py', 'w') as f:
    f.write(content)

print("Patched fast_test.py")
