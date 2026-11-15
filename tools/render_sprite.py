"""Render a canonical emoji splat (input view +Z, up +Y, unit size, centred) at a grid of views, RGBA, on a FIXED frame
(the unit box at --fill of the cell), so pruned or quantised copies line up pixel for pixel with the original.
Same rasteriser as render_pitch.py (each Gaussian its own projected ellipse, depth-sorted batches, orthographic).

usage: render_sprite.py SPLAT.ply|SPLAT.spz OUT.png --yaws -45,0,45 --pitches -25,0,25 [--cell 128] [--fill 0.84]
   writes one atlas: rows = pitches (camera above = +), columns = yaws (+ = camera to the emoji's right). With --npy
   also OUT.npy (float RGBA) for scoring. Yaw 0, pitch 0 = the input picture.
"""
import os, sys
import numpy as np
import cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spz

arg = lambda k, d: type(d)(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d
src, out = sys.argv[1], sys.argv[2]
yaws = [float(v) for v in arg("--yaws", "0").split(",")]
pitches = [float(v) for v in arg("--pitches", "0").split(",")]
H = W = arg("--cell", 128)
FILL = arg("--fill", 0.84)
BATCH = int(os.environ.get("SPLAT_BATCH", "400"))

a = spz.read(src) if src.endswith(".spz") else spz.read_ply(src)
P = np.stack([a["x"], a["y"], a["z"]], 1)
C = np.clip(0.5 + 0.28209479177387814 * np.stack([a["f_dc_0"], a["f_dc_1"], a["f_dc_2"]], 1), 0, 1)
op = 1 / (1 + np.exp(-a["opacity"]))
sc = np.exp(np.stack([a["scale_0"], a["scale_1"], a["scale_2"]], 1))
q = np.stack([a["rot_1"], a["rot_2"], a["rot_3"], a["rot_0"]], 1)
x, y, z, w = (q / np.linalg.norm(q, axis=1, keepdims=True)).T
Rg = np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], 1),
               np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], 1),
               np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], 1)], 1)
Sigma = np.einsum("nij,nj,nkj->nik", Rg, sc ** 2, Rg)
s = H * FILL                                     # the unit box (largest of width, height = 1) fills FILL of the cell


def draw(yaw, pitch):
    t, c = np.radians(yaw), np.radians(pitch)
    Ry = np.array([[np.cos(t), 0, -np.sin(t)], [0, 1, 0], [np.sin(t), 0, np.cos(t)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(c), -np.sin(c)], [0, np.sin(c), np.cos(c)]])
    R = Rx @ Ry
    Q = P @ R.T
    S2 = np.einsum("ij,njk,lk->nil", R, Sigma, R)[:, :2, :2] * s * s
    a_, b_, c_ = S2[:, 0, 0] + 0.3, -S2[:, 0, 1], S2[:, 1, 1] + 0.3
    det = np.maximum(a_ * c_ - b_ * b_, 1e-12)
    ia, ib, ic = c_ / det, -b_ / det, a_ / det
    rad = np.clip(np.ceil(3 * np.sqrt(np.maximum(a_, c_))).astype(int), 1, 14)
    u0, v0 = Q[:, 0] * s + W / 2, H / 2 - Q[:, 1] * s
    order = np.argsort(-Q[:, 2])
    img = np.zeros((H * W, 3)); Tm = np.ones(H * W)
    for i in range(0, len(order), BATCH):
        j = order[i:i + BATCH]
        if Tm.max() < 1e-3:
            break
        acc_c, acc_a, acc_l = np.zeros((H * W, 3)), np.zeros(H * W), np.zeros(H * W)
        for r in np.unique(rad[j]):
            k = j[rad[j] == r]
            d = np.arange(-r, r + 1)
            dx, dy = np.meshgrid(d, d); dx, dy = dx.ravel(), dy.ravel()
            px = np.round(u0[k]).astype(int)[:, None] + dx[None]
            py = np.round(v0[k]).astype(int)[:, None] + dy[None]
            ex, ey = px - u0[k][:, None], py - v0[k][:, None]
            g = np.exp(-0.5 * (ia[k][:, None] * ex * ex + 2 * ib[k][:, None] * ex * ey + ic[k][:, None] * ey * ey))
            al = np.clip(op[k][:, None] * g, 0, 0.99)
            ok = (px >= 0) & (px < W) & (py >= 0) & (py < H) & (al > 1 / 255)
            idx = (py * W + px)[ok]; av = al[ok]
            acc_a += np.bincount(idx, av, H * W); acc_l += np.bincount(idx, np.log1p(-av), H * W)
            ck = np.broadcast_to(C[k][:, None, :], (len(k), len(dx), 3))[ok]
            for ch in range(3):
                acc_c[:, ch] += np.bincount(idx, av * ck[:, ch], H * W)
        la = 1 - np.exp(acc_l)
        img += (Tm * la)[:, None] * (acc_c / np.maximum(acc_a, 1e-9)[:, None])
        Tm *= 1 - la
    alpha = 1 - Tm
    rgb = img / np.maximum(alpha, 1e-6)[:, None]       # un-premultiply for a straight-alpha PNG
    return np.concatenate([np.clip(rgb, 0, 1), alpha[:, None]], 1).reshape(H, W, 4)


atlas = np.vstack([np.hstack([draw(yw, p) for yw in yaws]) for p in pitches])
cv2.imwrite(out, (np.clip(atlas, 0, 1)[..., [2, 1, 0, 3]] * 255).round().astype(np.uint8))
if "--npy" in sys.argv:
    np.save(out.rsplit(".", 1)[0] + ".npy", atlas.astype(np.float32))
