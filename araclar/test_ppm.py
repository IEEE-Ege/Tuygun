import json
import math
import numpy as np

with open('mono_vo_data.json', 'r') as f:
    data = json.load(f)

ppm_x_list = []
ppm_y_list = []

# We need heading to do the rotation.
# Let's approximate heading from the data or just use a simplified approach.
# Since we just want to see if ppm_x and ppm_y differ significantly.
print("Veri yuklendi.")
