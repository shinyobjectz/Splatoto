"""Find the fixed rotation between Idol's .ply frame and the frame its predicted camera is written in.

For each of the 24 proper axis rotations Q and both signs of the camera vector, the splat (Q applied) is posed by the
camera (canon.py's frame, up +Z) and its front projected with a point z-buffer (solid splats, colour from f_dc); the
picture is scored against the source emoji: silhouette IoU - colour error, both cropped square. Prints, per candidate,
the score summed over all emoji, best first.

usage: orient_search.py CODE...   (reads modal/CODE.ply, modal/CODE_camera.json, in/3d/CODE.png)
"""
import itertools, json, sys
import numpy as np
from PIL import Image


def rotations():
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            m = np.zeros((3, 3))
            for i, (p, s) in enumerate(zip(perm, signs)):
                m[i, p] = s
            if np.linalg.det(m) > 0:
                out.append(m)
    return out


def load(code):
    with open(f"modal/{code}.ply", "rb") as fh:
        head = b""
        while not head.endswith(b"end_header\n"):
            head += fh.readline()
        names = [l.split()[-1] for l in head.decode().splitlines() if l.startswith("property")]
        n = int([l for l in head.decode().splitlines() if l.startswith("element vertex")][0].split()[-1])
        a = np.frombuffer(fh.read(), np.dtype([(k, "<f4") for k in names]), n)
    op = 1 / (1 + np.exp(-a["opacity"]))
    keep = op > 0.3
    P = np.stack([a["x"], a["y"], a["z"]], 1)[keep].astype(np.float64)
    C = np.clip(0.5 + 0.28209479 * np.stack([a["f_dc_0"], a["f_dc_1"], a["f_dc_2"]], 1)[keep], 0, 1)
    cam = json.load(open(f"modal/{code}_camera.json"))
    return P, C, cam


def frame(cam, sign):
    y, p = cam["yaw"], cam["pitch"]
    f = sign * np.array([np.cos(p) * np.sin(y), np.cos(p) * np.cos(y), np.sin(p)])
    u = np.array([0, 0, 1.0]) - f * f[2]; u /= np.linalg.norm(u)
    return np.stack([np.cross(u, f), u, f])


def square(rgb, mask, n=96):
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return np.zeros((n, n, 3)), np.zeros((n, n), bool)
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


def front(P, C, R, S=160):
    Q = P @ R.T
    lo, hi = Q[:, :2].min(0), Q[:, :2].max(0)
    k = (S - 1) / max(hi - lo)
    px = ((Q[:, 0] - lo[0]) * k).astype(int); py = ((hi[1] - Q[:, 1]) * k).astype(int)
    order = np.argsort(Q[:, 2])                     # far to near: the nearest write last
    img = np.zeros((S, S, 3)); m = np.zeros((S, S), bool)
    img[py[order], px[order]] = C[order]; m[py[order], px[order]] = True
    from scipy import ndimage as ndi
    m2 = ndi.binary_closing(m, iterations=2)
    return img, m2


codes = sys.argv[1:]
data = {c: load(c) for c in codes}
src = {}
for c in codes:
    s = np.asarray(Image.open(f"in/3d/{c}.png").convert("RGBA")) / 255
    src[c] = square(s[..., :3], s[..., 3] > 0.5)
rows = []
for qi, Qm in enumerate(rotations()):
    for sign in (1, -1):
        tot, per = 0.0, {}
        for c in codes:
            P, C, cam = data[c]
            img, m = front(P @ Qm.T, C, frame(cam, sign))
            r_rgb, r_m = square(img, m)
            s_rgb, s_m = src[c]
            inter, union = (s_m & r_m).sum(), (s_m | r_m).sum()
            iou = inter / union if union else 0
            col = np.abs(s_rgb[s_m & r_m] - r_rgb[s_m & r_m]).mean() if inter else 1
            per[c] = round(float(iou - col), 3); tot += iou - col
        rows.append((tot, qi, sign, Qm.astype(int).tolist(), per))
rows.sort(key=lambda r: -r[0])
for t, qi, sign, Qm, per in rows[:6]:
    print(round(t, 3), "Q", Qm, "sign", sign, per)
