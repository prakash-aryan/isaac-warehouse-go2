import json
import math
import sys
from pathlib import Path

import numpy as np

layout, dst = json.load(open(sys.argv[1])), Path(sys.argv[2])
res, margin = 0.05, 2.0
W, D = layout["size"]
w, h = int(round((W + 2 * margin) / res)), int(round((D + 2 * margin) / res))
img = np.full((h, w), 254, dtype=np.uint8)
for cx, cy, hx, hy, yaw in layout["footprints"]:
    c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    gx, gy = np.meshgrid(np.arange(-hx, hx + res / 2, res / 2), np.arange(-hy, hy + res / 2, res / 2))
    col = ((cx + c * gx - s * gy + margin) / res).astype(int)
    row = h - 1 - ((cy + s * gx + c * gy + margin) / res).astype(int)
    ok = (col >= 0) & (col < w) & (row >= 0) & (row < h)
    img[row[ok], col[ok]] = 0
dst.parent.mkdir(parents=True, exist_ok=True)
dst.with_suffix(".pgm").write_bytes(f"P5\n{w} {h}\n255\n".encode() + img.tobytes())
dst.with_suffix(".yaml").write_text(f"image: {dst.with_suffix('.pgm').name}\nmode: trinary\nresolution: {res}\n"
                                    f"origin: [{-margin}, {-margin}, 0.0]\nnegate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n")
print("map", w, "x", h, "cells,", int((img == 0).sum()), "occupied")
