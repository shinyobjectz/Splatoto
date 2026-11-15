"""Put an Idol .ply (PyTorch SDK, Z-up) in the emoji's canonical pose from Idol's own predicted input camera: the
input picture's view faces +Z, its up is +Y (glTF/three.js/raea: a camera on +Z looking down -Z sees the emoji as
drawn). Centred on its solid splats' box and scaled so the larger of width and height is 1.

idol/camera.py: direction d = (cos p sin yaw, cos p cos yaw, sin p), yaw = atan2(dx, dy), pitch = asin(dz). The frame:
f = s * d (toward the input camera; s = +1 unless --flip), u = +Z made perpendicular to f, r = u x f.

usage: canon.py IN.ply CAMERA.json OUT.ply [--flip]   prints the transform as JSON
"""
import json, sys
import numpy as np

src, cam_path, out = sys.argv[1:4]
cam = json.load(open(cam_path))
yaw, pitch = cam["yaw"], cam["pitch"]
d = np.array([np.cos(pitch) * np.sin(yaw), np.cos(pitch) * np.cos(yaw), np.sin(pitch)])
f = -d if "--flip" in sys.argv else d
_ax = sys.argv[sys.argv.index("--up") + 1] if "--up" in sys.argv else "+z"   # up in the camera's frame
up = np.zeros(3); up["xyz".index(_ax[-1])] = -1.0 if _ax[0] == "-" else 1.0
u = up - f * (f @ up)
u /= np.linalg.norm(u)
r = np.cross(u, f)
# the .ply's frame is not the camera's: orient_search.py over the 24 axis rotations on 8 emoji (11-15) found the camera
# frame = (x, z, -y) of the file (5.99 against 3.78 for the next), with d pointing toward the camera
Q = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0.0]])
R = np.stack([r, u, f]) @ Q                            # rows: canonical x, y, z in the file's frame

with open(src, "rb") as fh:
    head = b""
    while not head.endswith(b"end_header\n"):
        head += fh.readline()
    body = fh.read()
names = [l.split()[-1] for l in head.decode().splitlines() if l.startswith("property")]
n = int([l for l in head.decode().splitlines() if l.startswith("element vertex")][0].split()[-1])
a = np.frombuffer(body, np.dtype([(k, "<f4") for k in names]), n).copy()

P = np.stack([a["x"], a["y"], a["z"]], 1).astype(np.float64) @ R.T
op = 1 / (1 + np.exp(-a["opacity"].astype(np.float64)))
solid = op > 0.3
lo, hi = np.percentile(P[solid], 0.5, axis=0), np.percentile(P[solid], 99.5, axis=0)
centre = (lo + hi) / 2
k = 1.0 / max(hi[0] - lo[0], hi[1] - lo[1])
P = (P - centre) * k
a["x"], a["y"], a["z"] = P.T.astype(np.float32)
if "nx" in a.dtype.names:
    N = np.stack([a["nx"], a["ny"], a["nz"]], 1) @ R.T
    a["nx"], a["ny"], a["nz"] = N.T.astype(np.float32)
for i in range(3):
    a[f"scale_{i}"] = (a[f"scale_{i}"] + np.log(k)).astype(np.float32)   # log-scales: the uniform scale k


def mat_quat(m):
    t = np.trace(m)
    if t > 0:
        s = np.sqrt(t + 1) * 2; return np.array([s / 4, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    i = int(np.argmax(np.diag(m)))
    j, kk = (i + 1) % 3, (i + 2) % 3
    s = np.sqrt(1 + m[i, i] - m[j, j] - m[kk, kk]) * 2
    q = np.zeros(4); q[0] = (m[kk, j] - m[j, kk]) / s; q[1 + i] = s / 4
    q[1 + j] = (m[j, i] + m[i, j]) / s; q[1 + kk] = (m[kk, i] + m[i, kk]) / s
    return q


qw, qx, qy, qz = mat_quat(R)
w, x, y, z = (a[f"rot_{i}"].astype(np.float64) for i in range(4))   # 3DGS .ply: rot_0 = w; new = q_R * q
a["rot_0"] = (qw * w - qx * x - qy * y - qz * z).astype(np.float32)
a["rot_1"] = (qw * x + qx * w + qy * z - qz * y).astype(np.float32)
a["rot_2"] = (qw * y - qx * z + qy * w + qz * x).astype(np.float32)
a["rot_3"] = (qw * z + qx * y - qy * x + qz * w).astype(np.float32)
with open(out, "wb") as fh:
    fh.write(head.replace(b"end_header\n", b"comment noto3d canonical: input view +Z, up +Y, unit size\nend_header\n"))
    fh.write(a.tobytes())
print(json.dumps({"yaw": yaw, "pitch": pitch, "flip": "--flip" in sys.argv, "R": R.round(6).tolist(),
                  "centre": centre.round(6).tolist(), "scale": round(float(k), 6)}))
