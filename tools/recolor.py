"""Recolour an emoji splat by its colour groups, keeping its shading.

Each Gaussian's colour is split into groups by k-means in CIELAB (visible Gaussians only, weighted by opacity), so a
smiley falls into body yellow, rim orange, feature brown, teeth white and tongue red. A style moves groups in LCh:
the group's mean goes to a target, and every Gaussian keeps its offset from that mean (hue turned with it, chroma
scaled, lightness shifted), so rim shading and highlights carry over. Assignment is soft (Gaussian weights over the
group distances) so there are no seams where two groups blend.

    recolor.py IN.ply OUT.ply STYLE [ARG]      prints the groups as JSON
    import recolor; g = recolor.groups(a); a2 = recolor.apply(a, g, "tint", "#3a7bff")

Styles:
  tint HEX     the main (body) group becomes HEX; features, whites and darks stay
  mono HEX     every coloured group takes HEX's hue and chroma; lightness kept
  hue DEG      every group's hue turned by DEG
  pastel       lighter, half the chroma
  neon         chroma x1.9, darks deeper
  gold         lightness mapped onto a bronze-to-gold ramp, features kept darker
  chrome       no chroma, more contrast, a cool cast
  clay         chroma x0.55, lightness in 5 bands, warm
  ink          lightness in 3 bands, chroma x1.2 (a toon look)
  negative     lightness inverted, hue turned 180
"""
import json, sys
import numpy as np

SH_C0 = 0.28209479177387814


def srgb_to_lab(rgb):
    c = np.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    M = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]])
    xyz = c @ M.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])], 1)


def lab_to_srgb(lab):
    fy = (lab[:, 0] + 16) / 116; fx = fy + lab[:, 1] / 500; fz = fy - lab[:, 2] / 200
    f = np.stack([fx, fy, fz], 1)
    xyz = np.where(f ** 3 > 0.008856, f ** 3, (f - 16 / 116) / 7.787) * np.array([0.95047, 1.0, 1.08883])
    Mi = np.array([[3.2404542, -1.5371385, -0.4985314], [-0.9692660, 1.8760108, 0.0415560], [0.0556434, -0.2040259, 1.0572252]])
    c = xyz @ Mi.T
    return np.clip(np.where(c > 0.0031308, 1.055 * np.clip(c, 0, None) ** (1 / 2.4) - 0.055, 12.92 * c), 0, 1)


def hex_lab(h):
    h = h.lstrip("#")
    return srgb_to_lab(np.array([[int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]]))[0]


def colours(a):
    return np.clip(0.5 + SH_C0 * np.stack([a["f_dc_0"], a["f_dc_1"], a["f_dc_2"]], 1), 0, 1)


def groups(a, k=6, merge=14.0, seed=0):
    """k-means in Lab over the visible Gaussians; clusters closer than `merge` (delta E) are joined. Returns a dict
    with the centres (Lab), each group's visible share, and its role: body (largest coloured), feature (dark),
    white, accent."""
    lab = srgb_to_lab(colours(a))
    op = 1 / (1 + np.exp(-a["opacity"]))
    vis = np.nonzero(op > 0.3)[0]
    rng = np.random.default_rng(seed)
    pick = rng.choice(vis, min(30000, len(vis)), replace=False)
    X, w = lab[pick], op[pick]
    C = X[[rng.integers(len(X))]]
    for _ in range(1, k):                                 # k-means++ start
        d = ((X[:, None] - C[None]) ** 2).sum(2).min(1)
        C = np.vstack([C, X[rng.choice(len(X), p=d / d.sum())]])
    for _ in range(30):
        lab_i = ((X[:, None] - C[None]) ** 2).sum(2).argmin(1)
        C = np.array([np.average(X[lab_i == j], 0, w[lab_i == j]) if (lab_i == j).any() else C[j] for j in range(len(C))])
    while True:                                           # join near-duplicates
        D = np.sqrt(((C[:, None] - C[None]) ** 2).sum(2)) + np.eye(len(C)) * 1e9
        i, j = np.unravel_index(D.argmin(), D.shape)
        if D[i, j] > merge or len(C) <= 2:
            break
        lab_i = ((X[:, None] - C[None]) ** 2).sum(2).argmin(1)
        wi, wj = w[lab_i == i].sum(), w[lab_i == j].sum()
        C[i] = (C[i] * wi + C[j] * wj) / max(wi + wj, 1e-9); C = np.delete(C, j, 0)
    lab_i = ((X[:, None] - C[None]) ** 2).sum(2).argmin(1)
    share = np.array([w[lab_i == j].sum() for j in range(len(C))]); share /= share.sum()
    order = np.argsort(-share); C, share = C[order], share[order]
    chroma = np.hypot(C[:, 1], C[:, 2])
    roles = []
    for L, ch in zip(C[:, 0], chroma):
        roles.append("feature" if L < 32 else "white" if (L > 82 and ch < 18) else "accent")
    # the body is a hue family: coloured groups within 40 degrees of hue of each other, the family with the largest
    # visible share (a smiley's rim orange, mid yellow and highlight yellow are one body)
    hue = np.degrees(np.arctan2(C[:, 2], C[:, 1]))
    col = [j for j in range(len(C)) if roles[j] == "accent" and chroma[j] > 12]
    fam = {}
    for j in col:
        fam[j] = next((fam[i] for i in col if i in fam and abs((hue[i] - hue[j] + 180) % 360 - 180) < 40), j)
    if fam:
        best = max(set(fam.values()), key=lambda f: sum(share[j] for j in fam if fam[j] == f))
        for j in fam:
            if fam[j] == best:
                roles[j] = "body"
    else:
        roles[0] = "body"
    return {"centres": C, "share": share, "roles": roles}


def weights(lab, C, sigma=9.0):
    d2 = ((lab[:, None] - C[None]) ** 2).sum(2)
    d2 -= d2.min(1, keepdims=True)
    w = np.exp(-d2 / (2 * sigma ** 2))
    return w / w.sum(1, keepdims=True)


def lch(lab):
    return lab[:, 0], np.hypot(lab[:, 1], lab[:, 2]), np.arctan2(lab[:, 2], lab[:, 1])


def move(lab, m, t, chroma_from_target=True, hue_keep=0.35):
    """Every colour keeps its offset from the group mean m while the mean goes to t (Lab): hue turned by the
    difference, chroma scaled by the ratio, lightness shifted (and squeezed toward the ends)."""
    L, Cc, H = lch(lab)
    mL, mC, mH = m[0], np.hypot(m[1], m[2]), np.arctan2(m[2], m[1])
    tL, tC, tH = t[0], np.hypot(t[1], t[2]), np.arctan2(t[2], t[1])
    dL = L - mL
    nL = tL + dL * np.clip(np.where(dL > 0, (100 - tL) / max(100 - mL, 1), tL / max(mL, 1)), 0.3, 1.6)
    nC = Cc * (tC / max(mC, 1)) if chroma_from_target else Cc
    dH = (H - mH + np.pi) % (2 * np.pi) - np.pi           # each colour's hue offset from the mean, narrowed so a
    nH = tH + dH * hue_keep                                 # tint reads as one colour (yellow highlights don't go violet)
    return np.stack([nL, nC * np.cos(nH), nC * np.sin(nH)], 1)


def apply(a, g, style, arg=None):
    lab = srgb_to_lab(colours(a))
    C, roles = g["centres"], g["roles"]
    W = weights(lab, C)
    out = np.zeros_like(lab)
    L, Cc, H = lch(lab)
    body = [j for j in range(len(C)) if roles[j] == "body"]
    m_body = np.average(C[body], 0, g["share"][body])     # the body family moves as one
    for j in range(len(C)):
        m, r = C[j], roles[j]
        if style == "tint":
            new = move(lab, m_body, hex_lab(arg)) if r == "body" else lab
        elif style == "mono":
            tt = hex_lab(arg)
            if r in ("accent", "body"):
                k = np.hypot(tt[1], tt[2]) / max(np.hypot(m_body[1], m_body[2]), 1)
                new = np.stack([L, Cc * k * np.cos(np.arctan2(tt[2], tt[1])), Cc * k * np.sin(np.arctan2(tt[2], tt[1]))], 1)
            else:
                new = lab
        elif style == "hue":
            d = np.radians(float(arg))
            new = np.stack([L, Cc * np.cos(H + d), Cc * np.sin(H + d)], 1)
        elif style == "pastel":
            new = np.stack([L * 0.55 + 45, Cc * 0.5 * np.cos(H), Cc * 0.5 * np.sin(H)], 1)
        elif style == "neon":
            nL = np.where(L < 40, L * 0.6, L)
            new = np.stack([nL, np.minimum(Cc * 1.9, 128) * np.cos(H), np.minimum(Cc * 1.9, 128) * np.sin(H)], 1)
        elif style == "gold":
            x = np.clip(L / 100, 0, 1) ** 0.9
            ramp = np.array([[22, 12, 22], [48, 14, 45], [72, 6, 62], [92, -2, 48]])   # bronze to pale gold (Lab)
            pos = x * (len(ramp) - 1); i0 = np.clip(pos.astype(int), 0, len(ramp) - 2); f = (pos - i0)[:, None]
            new = ramp[i0] * (1 - f) + ramp[i0 + 1] * f
            if r == "feature":
                new = new * np.array([0.55, 0.8, 0.8])
        elif style == "chrome":
            nL = np.clip(50 + (L - 50) * 1.35, 0, 100)
            new = np.stack([nL, np.full_like(L, -2.0), np.full_like(L, -6.0)], 1)
        elif style == "clay":
            nL = np.round(L / 20) * 20 + 4
            new = np.stack([nL, Cc * 0.55 * np.cos(H) + 4, Cc * 0.55 * np.sin(H) + 8], 1)
        elif style == "ink":
            nL = np.select([L < 35, L < 70], [18, 58], 88)
            new = np.stack([nL, Cc * 1.2 * np.cos(H), Cc * 1.2 * np.sin(H)], 1)
        elif style == "negative":
            new = np.stack([100 - L, Cc * np.cos(H + np.pi), Cc * np.sin(H + np.pi)], 1)
        else:
            raise SystemExit(f"unknown style {style}")
        out += W[:, j:j + 1] * new
    rgb = lab_to_srgb(out)
    b = dict(a)
    f = (rgb - 0.5) / SH_C0
    b["f_dc_0"], b["f_dc_1"], b["f_dc_2"] = f[:, 0], f[:, 1], f[:, 2]
    return b


if __name__ == "__main__":
    sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
    import spz
    src, out, style = sys.argv[1:4]
    arg = sys.argv[4] if len(sys.argv) > 4 else None
    a = spz.read_ply(src)
    g = groups(a)
    (spz.write if out.endswith(".spz") else spz.write_ply)(apply(a, g, style, arg), out)
    print(json.dumps({"groups": [{"lab": c.round(1).tolist(), "share": round(float(s), 3), "role": r}
                                 for c, s, r in zip(g["centres"], g["share"], g["roles"])]}))
