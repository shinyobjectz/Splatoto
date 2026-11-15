"""Niantic .spz v3 (SH degree 0) writer and reader, byte-for-byte what raea's systems/splat/src/read.rs reads: gzip of
a 16-byte header (NGSP, version 3, count, degree 0, 12 fractional bits, flags, reserved), then per field for all
splats: positions (3 x 24-bit fixed point), alpha (sigmoid*255), colour (f_dc*0.15+0.5)*255, log-scale ((s+10)*16),
rotation (smallest three: largest's index in the top 2 bits, the other three as sign + 9-bit magnitude, x y z w).

    spz.py IN.ply OUT.spz [--keep N] [--rank opacity|contrib]   prints JSON (bytes, kept)
    import spz; a = spz.read_ply(p); spz.write(a, out); b = spz.read(out)   # b: the same field names, dequantised
"""
import gzip, json, struct, sys
import numpy as np

FIELDS = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2",
          "rot_0", "rot_1", "rot_2", "rot_3"]
FRAC = 12


def read_ply(path):
    with open(path, "rb") as f:
        head = b""
        while not head.endswith(b"end_header\n"):
            head += f.readline()
        names = [l.split()[-1] for l in head.decode().splitlines() if l.startswith("property")]
        n = int([l for l in head.decode().splitlines() if l.startswith("element vertex")][0].split()[-1])
        a = np.frombuffer(f.read(), np.dtype([(k, "<f4") for k in names]), n)
    return {k: a[k].astype(np.float64) for k in FIELDS}


def write_ply(a, path):
    n = len(a["x"])
    rec = np.zeros(n, np.dtype([(k, "<f4") for k in FIELDS]))
    for k in FIELDS:
        rec[k] = a[k]
    head = "ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % n + "".join(f"property float {k}\n" for k in FIELDS) + "end_header\n"
    with open(path, "wb") as f:
        f.write(head.encode()); f.write(rec.tobytes())


def rank(a, how="contrib"):
    """Indices, most visible first. contrib = opacity x projected area (the two largest axes), floored at a 1/512 pixel."""
    op = 1 / (1 + np.exp(-a["opacity"]))
    if how == "opacity":
        return np.argsort(-op)
    s = np.sort(np.exp(np.stack([a["scale_0"], a["scale_1"], a["scale_2"]], 1)), 1)
    return np.argsort(-(op * np.maximum(s[:, 2] * s[:, 1], (1 / 512) ** 2)))


def subset(a, idx):
    return {k: v[idx] for k, v in a.items()}


def encode(a):
    n = len(a["x"])
    pos = np.round(np.stack([a["x"], a["y"], a["z"]], 1) * (1 << FRAC)).astype(np.int64)
    pos = np.clip(pos, -(1 << 23), (1 << 23) - 1) & 0xFFFFFF
    pos_b = np.stack([pos & 0xFF, (pos >> 8) & 0xFF, (pos >> 16) & 0xFF], -1).astype(np.uint8).reshape(n, 9)
    alpha = np.clip(np.round(255 / (1 + np.exp(-a["opacity"]))), 0, 255).astype(np.uint8)
    col = np.clip(np.round((np.stack([a["f_dc_0"], a["f_dc_1"], a["f_dc_2"]], 1) * 0.15 + 0.5) * 255), 0, 255).astype(np.uint8)
    scl = np.clip(np.round((np.stack([a["scale_0"], a["scale_1"], a["scale_2"]], 1) + 10) * 16), 0, 255).astype(np.uint8)
    q = np.stack([a["rot_1"], a["rot_2"], a["rot_3"], a["rot_0"]], 1)          # x y z w
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    big = np.argmax(np.abs(q), 1)
    q *= np.where(q[np.arange(n), big] < 0, -1.0, 1.0)[:, None]
    v = np.zeros(n, np.int64)
    for c in range(4):
        m = big != c
        mag = np.clip(np.round(np.abs(q[:, c]) / np.sqrt(0.5) * 511), 0, 511).astype(np.int64)
        comp = ((q[:, c] < 0).astype(np.int64) << 9) | mag
        v = np.where(m, (v << 10) | comp, v)
    v |= big.astype(np.int64) << 30
    rot_b = v.astype("<u4").view(np.uint8).reshape(n, 4)
    head = b"NGSP" + struct.pack("<II", 3, n) + bytes([0, FRAC, 0, 0])
    return head + pos_b.tobytes() + alpha.tobytes() + col.tobytes() + scl.tobytes() + rot_b.tobytes()


def write(a, path, level=9):
    with open(path, "wb") as f:
        f.write(gzip.compress(encode(a), compresslevel=level, mtime=0))


def read(path):
    """Mirror of read.rs: what raea will see, back in .ply units (log-scale, logit opacity, f_dc)."""
    d = gzip.decompress(open(path, "rb").read())
    assert d[:4] == b"NGSP"
    ver, n = struct.unpack("<II", d[4:12]); deg, frac = d[12], d[13]
    assert ver == 3 and deg == 0
    o = 16
    p = np.frombuffer(d, np.uint8, n * 9, o).reshape(n, 3, 3).astype(np.int64); o += n * 9
    x = p[..., 0] | p[..., 1] << 8 | p[..., 2] << 16
    x = np.where(x & 0x800000, x - (1 << 24), x) / (1 << frac)
    al = np.frombuffer(d, np.uint8, n, o) / 255; o += n
    col = np.frombuffer(d, np.uint8, n * 3, o).reshape(n, 3) / 255; o += n * 3
    scl = np.frombuffer(d, np.uint8, n * 3, o).reshape(n, 3) / 16 - 10; o += n * 3
    v = np.frombuffer(d, "<u4", n, o).astype(np.int64)
    big = v >> 30
    q = np.zeros((n, 4)); ss = np.zeros(n); bits = v.copy()
    for c in (3, 2, 1, 0):
        m = big != c
        mag = (bits & 0x1FF) / 511 * np.sqrt(0.5); neg = (bits >> 9) & 1
        q[:, c] = np.where(m, np.where(neg == 1, -mag, mag), q[:, c]); ss += np.where(m, q[:, c] ** 2, 0)
        bits = np.where(m, bits >> 10, bits)
    q[np.arange(n), big] = np.sqrt(np.maximum(1 - ss, 0))
    al = np.clip(al, 1e-6, 1 - 1e-6)
    return {"x": x[:, 0], "y": x[:, 1], "z": x[:, 2], "f_dc_0": (col[:, 0] - 0.5) / 0.15, "f_dc_1": (col[:, 1] - 0.5) / 0.15,
            "f_dc_2": (col[:, 2] - 0.5) / 0.15, "opacity": np.log(al / (1 - al)), "scale_0": scl[:, 0], "scale_1": scl[:, 1],
            "scale_2": scl[:, 2], "rot_0": q[:, 3], "rot_1": q[:, 0], "rot_2": q[:, 1], "rot_3": q[:, 2]}


if __name__ == "__main__":
    src, out = sys.argv[1], sys.argv[2]
    a = read_ply(src)
    keep = int(sys.argv[sys.argv.index("--keep") + 1]) if "--keep" in sys.argv else len(a["x"])
    how = sys.argv[sys.argv.index("--rank") + 1] if "--rank" in sys.argv else "contrib"
    a = subset(a, np.sort(rank(a, how)[:keep]))
    write(a, out)
    import os
    print(json.dumps({"out": out, "kept": keep, "bytes": os.path.getsize(out)}))
