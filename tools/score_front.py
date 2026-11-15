"""Score a canonical splat's front (camera on +Z) against its source emoji: silhouette IoU and mean colour difference
inside the shared silhouette, both cropped to the object and padded square (fit_view.py's measure).

usage: score_front.py SPLAT.ply SOURCE.png [OUT.jpg]   prints JSON
"""
import json, os, subprocess, sys, tempfile
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.expanduser("~/Documents/raea/internal/projects/sheet3d/.venv/bin/python")
splat, source = sys.argv[1], sys.argv[2]
out = sys.argv[3] if len(sys.argv) > 3 else tempfile.mktemp(suffix=".jpg")
BG = 0.16
subprocess.run([PY, f"{HERE}/render_pitch.py", splat, out, "0", "--height", "256", "--bg", str(BG)],
               env={**os.environ, "PITCH": "0"}, check=True, capture_output=True)


def square(rgb, mask, n=128):
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    side = max(y1 - y0, x1 - x0); cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    o_rgb, o_m = np.zeros((side, side, 3)), np.zeros((side, side))
    sy, sx = cy - side // 2, cx - side // 2
    for arr, dst in ((rgb, o_rgb), (mask[..., None].astype(float), o_m[..., None])):
        ya, yb = max(sy, 0), min(sy + side, arr.shape[0]); xa, xb = max(sx, 0), min(sx + side, arr.shape[1])
        dst[ya - sy:yb - sy, xa - sx:xb - sx] = arr[ya:yb, xa:xb]
    r = np.asarray(Image.fromarray((o_rgb * 255).astype(np.uint8)).resize((n, n), Image.BILINEAR)) / 255
    m = np.asarray(Image.fromarray((o_m * 255).astype(np.uint8)).resize((n, n), Image.BILINEAR)) / 255 > 0.5
    return r, m


s = np.asarray(Image.open(source).convert("RGBA")) / 255
s_rgb, s_m = square(s[..., :3], s[..., 3] > 0.5)
im = np.asarray(Image.open(out).convert("RGB")) / 255
r_rgb, r_m = square(im, np.abs(im - BG).max(2) > 0.06)
inter, union = (s_m & r_m).sum(), (s_m | r_m).sum()
col = float(np.abs(s_rgb[s_m & r_m] - r_rgb[s_m & r_m]).mean()) if inter else 1.0
print(json.dumps({"splat": os.path.basename(splat), "iou": round(float(inter / union), 4), "colour_err": round(col, 4)}))
