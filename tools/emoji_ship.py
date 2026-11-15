"""Ship step for the Noto 3D -> Idol library, on Modal.

    modal run emoji_ship.py --codes 1f600,1f680 [--containers 10] [--upload]
    modal run emoji_ship.py --list codes.txt [--containers 20] [--upload]

Per emoji, from /vol/emoji/out/CODE.ply + CODE_camera.json (emoji_modal.py's output) into /vol/emoji/ship/:
  CODE.webp        the cursor cone: 9 yaws (-45..45 by 11.25) x 5 pitches (+25..-25 by 12.5), 128 px cells, RGBA
  CODE_360.webp    a full turn: 32 yaws (0..348.75 by 11.25) x the same 5 pitches; cone = its columns 28..31, 0..4
  CODE_front.webp  the input view alone at 256 px
  CODE_parts.webp, CODE_360_parts.webp   the same cells as the two atlases, RGB = how much of each pixel is the body
                   (R), dark features (G) and whites (B); the rest is accent. For live recolouring (recolor.js)
  CODE.spz         every Gaussian (262,144), quantised (spz.py, what raea's read.rs reads); ~3.4 MB
  CODE_65k.spz     the 65,536 most visible (opacity x projected area); ~0.9 MB
  CODE.json        pose, front score against the source picture, back check, colour groups (recolor.py)
Canonical pose (canon.py): the input view faces +Z, up +Y, centred on the solid splats' box, largest of w/h = 1.
Views: yaw + = the camera moves to the picture's right (the emoji turns to the picture's left); pitch + = camera above.
Rendered with gsplat (orthographic, classic 3DGS rasterisation, eps2d 0.3), straight alpha.
With --upload the two .spz go to an S3-compatible bucket (env SPLAT_BUCKET, Modal secret splats-r2).
"""
import json, os, subprocess, time
import modal

HERE = os.path.dirname(os.path.abspath(__file__))
vol = modal.Volume.from_name("idol-work")
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("webp")
    .pip_install("torch==2.4.0", index_url="https://download.pytorch.org/whl/cu124")
    .pip_install("gsplat==1.5.3", extra_index_url="https://docs.gsplat.studio/whl/pt24cu124")
    .pip_install("numpy<2", "pillow>=10", "boto3", "packaging", "jaxtyping", "rich")
    .add_local_file(f"{HERE}/spz.py", "/root/spz.py")
    .add_local_file(f"{HERE}/recolor.py", "/root/recolor.py")
    .add_local_file(f"{HERE}/OFL.txt", "/root/OFL.txt")
)
app = modal.App("raea-emoji3d-ship", image=image)

CONE_YAWS = [-45 + 11.25 * i for i in range(9)]
TURN_YAWS = [11.25 * i for i in range(32)]
PITCHES = [25, 12.5, 0, -12.5, -25]
CELL, FILL = 128, 0.84
Q_FILE = [[1, 0, 0], [0, 0, 1], [0, -1, 0.0]]
# the raea app's project hues (--a0..--a7 = oklch(0.78 0.13 h), h 25..345): "tint" looks pre-rendered for the marks
TINTS = {"a0": "#ff958d", "a1": "#eca851", "a2": "#b5c159", "a3": "#61d19a", "a4": "#1acfdf", "a5": "#73bdff",
         "a6": "#c4a4fe", "a7": "#f294ca"}


def back_ok(back, body_lab):
    """The back fails only where Idol carries a yellow or skin-toned front's rim shading round (smileys, hands,
    people): back lightness under 0.72 of the front's AND a body hue of 35-100 degrees with chroma over 20. Dark
    backs on other things (a controller, a clapperboard) are real. Measured on 21 emoji, 11-15."""
    import math
    h = math.degrees(math.atan2(body_lab[2], body_lab[1])) % 360
    skin = 35 <= h <= 100 and math.hypot(body_lab[1], body_lab[2]) > 20
    return bool(back["L_ratio"] >= 0.72 or not skin)   # camera frame = (x, z, -y) of Idol's .ply (orient_search.py, 11-15)


def canon(a, cam):
    import numpy as np
    yaw, pitch = cam["yaw"], cam["pitch"]
    f = np.array([np.cos(pitch) * np.sin(yaw), np.cos(pitch) * np.cos(yaw), np.sin(pitch)])
    up = np.array([0, 0, 1.0]); u = up - f * (f @ up); u /= np.linalg.norm(u); r = np.cross(u, f)
    R = np.stack([r, u, f]) @ np.array(Q_FILE)
    P = np.stack([a["x"], a["y"], a["z"]], 1) @ R.T
    op = 1 / (1 + np.exp(-a["opacity"]))
    solid = op > 0.3
    lo, hi = np.percentile(P[solid], 0.5, axis=0), np.percentile(P[solid], 99.5, axis=0)
    centre = (lo + hi) / 2; k = 1.0 / max(hi[0] - lo[0], hi[1] - lo[1])
    P = (P - centre) * k
    a = dict(a); a["x"], a["y"], a["z"] = P.T
    for i in range(3):
        a[f"scale_{i}"] = a[f"scale_{i}"] + np.log(k)
    # rotations: q_R * q (3DGS .ply rot_0 = w)
    m = R; t = np.trace(m)
    if t > 0:
        s = np.sqrt(t + 1) * 2; qw, qx, qy, qz = s / 4, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    else:
        i = int(np.argmax(np.diag(m))); j, kk = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(1 + m[i, i] - m[j, j] - m[kk, kk]) * 2
        q = np.zeros(4); q[0] = (m[kk, j] - m[j, kk]) / s; q[1 + i] = s / 4
        q[1 + j] = (m[j, i] + m[i, j]) / s; q[1 + kk] = (m[kk, i] + m[i, kk]) / s
        qw, qx, qy, qz = q
    w, x, y, z = (a[f"rot_{i}"] for i in range(4))
    a["rot_0"] = qw * w - qx * x - qy * y - qz * z
    a["rot_1"] = qw * x + qx * w + qy * z - qz * y
    a["rot_2"] = qw * y - qx * z + qy * w + qz * x
    a["rot_3"] = qw * z + qx * y - qy * x + qz * w
    return a, {"R": R.round(6).tolist(), "centre": centre.round(6).tolist(), "scale": float(k)}


class Renderer:
    def __init__(self, a):
        import numpy as np, torch
        d = "cuda"
        self.t = lambda v: torch.tensor(np.asarray(v), dtype=torch.float32, device=d)
        self.means = self.t(np.stack([a["x"], a["y"], a["z"]], 1))
        self.quats = self.t(np.stack([a["rot_0"], a["rot_1"], a["rot_2"], a["rot_3"]], 1))
        self.scales = self.t(np.exp(np.stack([a["scale_0"], a["scale_1"], a["scale_2"]], 1)))
        self.op = self.t(1 / (1 + np.exp(-a["opacity"])))
        self.col = self.t(np.clip(0.5 + 0.28209479177387814 * np.stack([a["f_dc_0"], a["f_dc_1"], a["f_dc_2"]], 1), 0, 1))

    def views(self, cells, size, fill=FILL, chunk=40, colors=None):
        """cells: [(yaw, pitch)] in degrees -> float32 [n, size, size, 4] straight-alpha RGBA. colors: [N, 3] in
        place of the splat's own (the parts weights)."""
        import numpy as np, torch
        from gsplat import rasterization
        s = size * fill
        out = []
        for i in range(0, len(cells), chunk):
            vm = []
            for yaw, pitch in cells[i:i + chunk]:
                t, c = np.radians(yaw), np.radians(pitch)
                Ry = np.array([[np.cos(t), 0, -np.sin(t)], [0, 1, 0], [np.sin(t), 0, np.cos(t)]])
                Rx = np.array([[1, 0, 0], [0, np.cos(c), -np.sin(c)], [0, np.sin(c), np.cos(c)]])
                M = np.eye(4); M[:3, :3] = np.diag([1, -1, -1]) @ Rx @ Ry; M[2, 3] = 10.0   # OpenCV camera, 10 units back
                vm.append(M)
            n = len(vm)
            K = self.t(np.array([[s, 0, size / 2], [0, s, size / 2], [0, 0, 1]]))[None].repeat(n, 1, 1)
            rgb, alpha, _ = rasterization(self.means, self.quats, self.scales, self.op, self.col if colors is None else colors,
                                          self.t(np.stack(vm)), K,
                                          size, size, camera_model="ortho", near_plane=0.01, far_plane=100.0, eps2d=0.3,
                                          render_mode="RGB", packed=False)
            rgb = rgb / alpha.clamp_min(1e-6)
            out.append(torch.cat([rgb.clamp(0, 1), alpha.clamp(0, 1)], -1).cpu().numpy())
        return np.concatenate(out)


def atlas(cells, cols):
    import numpy as np
    rows = [np.concatenate(list(cells[r * cols:(r + 1) * cols]), 1) for r in range(len(cells) // cols)]
    return np.concatenate(rows, 0)


def webp(arr, path, q=80, aq=90):
    import numpy as np
    from PIL import Image
    png = path + ".png"
    Image.fromarray((np.clip(arr, 0, 1) * 255).round().astype("uint8"), "RGBA").save(png)
    subprocess.run(["cwebp", "-quiet", "-q", str(q), "-alpha_q", str(aq), "-m", "6", "-exact", png, "-o", path], check=True)
    os.remove(png)


def square(rgb, mask, n=128):
    import numpy as np
    from PIL import Image
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
    r = np.asarray(Image.fromarray((o_rgb * 255).astype("uint8")).resize((n, n), Image.BILINEAR)) / 255
    m = np.asarray(Image.fromarray((o_m * 255).astype("uint8")).resize((n, n), Image.BILINEAR)) / 255 > 0.5
    return r, m


def lab_L(rgb):
    import numpy as np
    c = np.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    Y = c @ np.array([0.2126, 0.7152, 0.0722])
    return np.where(Y > 0.008856, 116 * np.cbrt(Y) - 16, 903.3 * Y)


@app.cls(gpu=["L4", "A10G", "T4"], volumes={"/vol": vol}, timeout=1800, memory=16384, max_containers=10,
         secrets=[modal.Secret.from_name("splats-r2", required_keys=["R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_ENDPOINT"])])
class Ship:
    @modal.method()
    def run(self, code: str, upload: bool = False, force: bool = False, tints: bool = True, retint: bool = False):
        import sys
        sys.path.insert(0, "/root")
        import numpy as np
        from PIL import Image
        import spz
        src, dst = "/vol/emoji/out", "/vol/emoji/ship"
        os.makedirs(dst, exist_ok=True)
        jpath = f"{dst}/{code}.json"
        have = os.path.exists(jpath) and not force
        tint_have = not retint and all(os.path.exists(f"{dst}/tint/{n}/{code}.webp") for n in TINTS)
        if have and (tint_have or not tints):
            return {"code": code, "skipped": True}
        if not os.path.exists(f"{src}/{code}.ply"):
            vol.reload()
            if not os.path.exists(f"{src}/{code}.ply"):
                return {"code": code, "missing": True}
        t0 = time.time()
        a = spz.read_ply(f"{src}/{code}.ply")
        cam = json.load(open(f"{src}/{code}_camera.json"))
        a, pose = canon(a, cam)
        import recolor
        r = Renderer(a)
        g = recolor.groups(a)
        info = json.load(open(jpath)) if have else self.full(code, a, cam, pose, r, g, upload, t0)
        if tints and (force or not tint_have):
            cells = [(y, p) for p in PITCHES for y in CONE_YAWS]
            for n, hexc in TINTS.items():
                os.makedirs(f"{dst}/tint/{n}", exist_ok=True)
                v = r.views(cells, CELL, colors=r.t(recolor.colours(recolor.apply(a, g, "tint", hexc))))
                webp(atlas(v, 9), f"{dst}/tint/{n}/{code}.webp")
            info["tints"] = sorted(TINTS)
            info["seconds"] = round(time.time() - t0, 2)
        json.dump(info, open(jpath, "w"))
        return info

    def full(self, code, a, cam, pose, r, g, upload, t0):
        import numpy as np
        from PIL import Image
        import spz, recolor
        dst = "/vol/emoji/ship"
        spz.write(a, f"{dst}/{code}.spz")
        small = spz.subset(a, np.sort(spz.rank(a)[:65536]))
        spz.write(small, f"{dst}/{code}_65k.spz")
        turn = r.views([(y, p) for p in PITCHES for y in TURN_YAWS], CELL)
        webp(atlas(turn, 32), f"{dst}/{code}_360.webp")
        cone_idx = [(round(y / 11.25) % 32) for y in CONE_YAWS]
        cone = [turn[pi * 32 + yi] for pi in range(5) for yi in cone_idx]
        webp(atlas(cone, 9), f"{dst}/{code}.webp")
        front = r.views([(0, 0)], 256)[0]
        webp(front, f"{dst}/{code}_front.webp", q=85)
        # parts for live recolouring: each Gaussian's soft weights over its colour groups, summed by role
        Wg = recolor.weights(recolor.srgb_to_lab(recolor.colours(a)), g["centres"])
        role = np.array(g["roles"])
        parts = np.stack([Wg[:, role == k].sum(1) for k in ("body", "feature", "white")], 1)
        pt = r.views([(y, p) for p in PITCHES for y in TURN_YAWS], CELL, colors=r.t(parts))
        pt = np.concatenate([pt[..., :3] * (pt[..., 3:] > 0.004), np.ones_like(pt[..., 3:])], -1)
        webp(atlas(pt, 32), f"{dst}/{code}_360_parts.webp", q=85, aq=100)
        webp(atlas([pt[pi * 32 + yi] for pi in range(5) for yi in cone_idx], 9), f"{dst}/{code}_parts.webp", q=85, aq=100)
        body = [j for j in range(len(role)) if role[j] == "body"]
        groups_info = {"centres_lab": g["centres"].round(2).tolist(), "share": g["share"].round(4).tolist(), "roles": g["roles"],
                       "body_lab": np.average(g["centres"][body], 0, g["share"][body]).round(2).tolist()}
        # front score against the source picture (score_front.py's measure)
        s = np.asarray(Image.open(f"/vol/emoji/in/{code}.png").convert("RGBA")) / 255
        s_rgb, s_m = square(s[..., :3], s[..., 3] > 0.5)
        f_rgb, f_m = square(front[..., :3], front[..., 3] > 0.5)
        inter, union = (s_m & f_m).sum(), (s_m | f_m).sum()
        col = float(np.abs(s_rgb[s_m & f_m] - f_rgb[s_m & f_m]).mean()) if inter else 1.0
        # back check: median lightness of the back cone (yaw 168.75..191.25) over the front cone's, pitch 0 row
        row0 = turn[2 * 32:3 * 32]
        px = lambda ids: np.concatenate([row0[i][row0[i][..., 3] > 0.9][:, :3] for i in ids])
        F, B = lab_L(px([31, 0, 1])), lab_L(px([15, 16, 17]))
        ratio = float(np.median(B) / max(np.median(F), 1e-6))
        dark = float((B < np.percentile(F, 25)).mean())
        info = {"code": code, "camera": cam, "pose": pose, "front": {"iou": round(float(inter / union), 4), "colour_err": round(col, 4)},
                "back": {"L_ratio": round(ratio, 3), "dark": round(dark, 3), "ok": back_ok({"L_ratio": ratio}, groups_info["body_lab"])},
                "groups": groups_info,
                "bytes": {k: os.path.getsize(f"{dst}/{code}{k}") for k in (".webp", "_360.webp", "_front.webp", "_parts.webp", "_360_parts.webp", ".spz", "_65k.spz")},
                "seconds": round(time.time() - t0, 2)}
        if upload:
            import boto3
            s3 = boto3.client("s3", endpoint_url=os.environ["R2_ENDPOINT"], aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
                              aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"], region_name="auto")
            for k in (".spz", "_65k.spz"):
                s3.upload_file(f"{dst}/{code}{k}", os.environ.get("SPLAT_BUCKET", "splats"), f"emoji3d/v1/{code}{k}", ExtraArgs={"ContentType": "application/octet-stream"})
            info["uploaded"] = True
        return info

    @modal.exit()
    def flush(self):
        vol.commit()


SPRITES = (".webp", "_parts.webp", "_front.webp")
TURN = ("_360.webp", "_360_parts.webp")


@app.function(volumes={"/vol": vol}, timeout=3600, cpu=4, memory=8192,
              secrets=[modal.Secret.from_name("media-r2", required_keys=["MEDIA_R2_ACCESS_KEY_ID", "MEDIA_R2_SECRET_ACCESS_KEY", "MEDIA_R2_ENDPOINT"])])
def pack(meta: dict, upload: bool = False, version: str = "v1") -> dict:
    """index.json over every shipped emoji, one zip per Unicode group in /vol/emoji/pack/ (the GitHub release), and
    with upload the sprites + index.json to the bucket MEDIA_BUCKET under emoji3d/<version>/ (public)."""
    import glob, re, zipfile
    from concurrent.futures import ThreadPoolExecutor
    vol.reload()
    ship, out = "/vol/emoji/ship", "/vol/emoji/pack"
    os.makedirs(out, exist_ok=True)
    index = {"version": version, "tint_hues": TINTS, "cell": CELL, "fill": FILL, "pitches": PITCHES, "cone_yaws": CONE_YAWS, "turn_yaws": TURN_YAWS,
             "source": "googlefonts/noto-emoji 3D/png/512 @ 99701e0", "licence": "OFL-1.1", "emoji": {}}
    for j in sorted(glob.glob(f"{ship}/*.json")):
        info = json.load(open(j)); code = info["code"]
        m = meta.get(code, {"group": "Other", "subgroup": "other", "name": code, "char": ""})
        index["emoji"][code] = {"name": m["name"], "char": m["char"], "group": m["group"], "subgroup": m["subgroup"],
                                "has360": back_ok(info["back"], info["groups"]["body_lab"]),
                                "tints": all(os.path.exists(f"{ship}/tint/{n}/{code}.webp") for n in TINTS), "front_iou": info["front"]["iou"], "front_colour_err": info["front"]["colour_err"],
                                "body_lab": info["groups"]["body_lab"]}
    files = lambda c, e: list(SPRITES) + (list(TURN) if e["has360"] else [])
    slug = lambda g: re.sub(r"[^a-z0-9]+", "-", g.lower()).strip("-")
    groups = {}
    for c, e in index["emoji"].items():
        groups.setdefault(e["group"], []).append(c)
    idx_bytes = json.dumps(index, ensure_ascii=False, separators=(",", ":")).encode()
    open(f"{out}/index.json", "wb").write(idx_bytes)
    zips = {}
    for g, codes in groups.items():
        zp = f"{out}/splatoto-{version}-{slug(g)}.zip"
        with zipfile.ZipFile(zp + ".part", "w", zipfile.ZIP_STORED, strict_timestamps=False) as z:
            z.write("/root/OFL.txt", "OFL.txt")
            sub = {**index, "emoji": {c: index["emoji"][c] for c in codes}}
            z.writestr("index.json", json.dumps(sub, ensure_ascii=False, indent=0))
            for c in codes:
                for k in files(c, index["emoji"][c]):
                    z.write(f"{ship}/{c}{k}", f"{c}{k}")
        os.replace(zp + ".part", zp)
        zips[os.path.basename(zp)] = {"emoji": len(codes), "bytes": os.path.getsize(zp)}
    vol.commit()
    res = {"emoji": len(index["emoji"]), "has360": sum(e["has360"] for e in index["emoji"].values()), "zips": zips}
    if upload:
        import boto3
        def put(item):
            key, path, ctype, cache = item
            s3 = boto3.client("s3", endpoint_url=os.environ["MEDIA_R2_ENDPOINT"], aws_access_key_id=os.environ["MEDIA_R2_ACCESS_KEY_ID"],
                              aws_secret_access_key=os.environ["MEDIA_R2_SECRET_ACCESS_KEY"], region_name="auto")
            s3.upload_file(path, os.environ.get("MEDIA_BUCKET", "media"), key, ExtraArgs={"ContentType": ctype, "CacheControl": cache})
        items = [(f"emoji3d/{version}/{c}{k}", f"{ship}/{c}{k}", "image/webp", "public, max-age=31536000, immutable")
                 for c, e in index["emoji"].items() for k in files(c, e)]
        items += [(f"emoji3d/{version}/tint/{n}/{c}.webp", f"{ship}/tint/{n}/{c}.webp", "image/webp", "public, max-age=31536000, immutable")
                  for c, e in index["emoji"].items() if e["tints"] for n in TINTS]
        with ThreadPoolExecutor(32) as ex:
            list(ex.map(put, items))
        put((f"emoji3d/{version}/index.json", f"{out}/index.json", "application/json", "public, max-age=300"))
        res["uploaded"] = len(items) + 1
    return res


@app.local_entrypoint()
def main(codes: str = "", list: str = "", containers: int = 10, upload: bool = False, force: bool = False,
         do_pack: bool = False, retint: bool = False):
    if do_pack:
        meta = json.load(open("codes_meta.json"))
        print(json.dumps(pack.remote(meta, upload=upload), indent=1))
        return
    todo = [c for c in codes.split(",") if c] or [l.strip() for l in open(list) if l.strip()]
    t0 = time.time()
    cls = Ship.with_options(max_containers=containers)
    with open("ship_runs.jsonl", "a") as log:
        for r in cls().run.map(todo, kwargs={"upload": upload, "force": force, "retint": retint}, return_exceptions=True):
            r = r if isinstance(r, dict) else {"error": repr(r)[:400]}
            log.write(json.dumps(r) + "\n"); log.flush()
            print(json.dumps({k: r.get(k) for k in ("code", "front", "back", "seconds", "error", "skipped", "missing")})[:300])
    print(f"== {len(todo)} in {time.time() - t0:.0f}s")
