// Live recolouring of a Splatoto sprite atlas, using its parts atlas (<code>_parts.webp / <code>_360_parts.webp).
// The same looks as tools/recolor.py, which recolours the splat itself; this works on finished pictures instead.
//
//   import { recolor, parseLook } from "./recolor.js";
//   const out = recolor(atlasImageData, partsImageData, parseLook("tint #3a7bff"));   // a new ImageData
//
// Parts atlas: per pixel, R = how much is the body (the main colour family), G = dark features, B = whites; the rest
// is accent. Looks move colours in CIELAB/LCh and keep each pixel's offset from its group, so shading survives.
//
// Looks: "tint #hex" (body only) · "mono #hex" (every coloured part takes the hue) · "hue DEG" · "pastel" · "neon" ·
//        "gold" · "chrome" · "clay" · "ink" · "negative"

export function parseLook(text) {
  const [style, arg] = String(text || "").trim().split(/\s+/);
  return { style: (style || "").toLowerCase(), arg };
}

// sRGB (0..1) <-> CIELAB, D65
const lin = (c) => (c > 0.04045 ? ((c + 0.055) / 1.055) ** 2.4 : c / 12.92);
const gam = (c) => (c > 0.0031308 ? 1.055 * Math.max(c, 0) ** (1 / 2.4) - 0.055 : 12.92 * c);
const fwd = (t) => (t > 0.008856 ? Math.cbrt(t) : 7.787 * t + 16 / 116);
const inv = (f) => (f ** 3 > 0.008856 ? f ** 3 : (f - 16 / 116) / 7.787);

export function toLab(r, g, b) {
  const R = lin(r), G = lin(g), B = lin(b);
  const x = (0.4124564 * R + 0.3575761 * G + 0.1804375 * B) / 0.95047;
  const y = 0.2126729 * R + 0.7151522 * G + 0.072175 * B;
  const z = (0.0193339 * R + 0.119192 * G + 0.9503041 * B) / 1.08883;
  const fx = fwd(x), fy = fwd(y), fz = fwd(z);
  return [116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)];
}

export function toRgb(L, a, b) {
  const fy = (L + 16) / 116, fx = fy + a / 500, fz = fy - b / 200;
  const x = inv(fx) * 0.95047, y = inv(fy), z = inv(fz) * 1.08883;
  const R = 3.2404542 * x - 1.5371385 * y - 0.4985314 * z;
  const G = -0.969266 * x + 1.8760108 * y + 0.041556 * z;
  const B = 0.0556434 * x - 0.2040259 * y + 1.0572252 * z;
  return [gam(R), gam(G), gam(B)].map((v) => Math.min(1, Math.max(0, v)));
}

export function hexLab(hex) {
  const h = String(hex || "#888888").replace("#", "");
  return toLab(parseInt(h.slice(0, 2), 16) / 255, parseInt(h.slice(2, 4), 16) / 255, parseInt(h.slice(4, 6), 16) / 255);
}

const wrap = (h) => ((((h + Math.PI) % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI)) - Math.PI;
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const polar = (L, C, H) => [L, C * Math.cos(H), C * Math.sin(H)];

// the group mean m goes to t; each colour keeps its offset (lightness squeezed toward the ends, chroma scaled, hue
// offset narrowed so a tint reads as one colour)
function move(L, C, H, m, t) {
  const mC = Math.hypot(m[1], m[2]), mH = Math.atan2(m[2], m[1]);
  const tC = Math.hypot(t[1], t[2]), tH = Math.atan2(t[2], t[1]);
  const dL = L - m[0];
  const k = dL > 0 ? (100 - t[0]) / Math.max(100 - m[0], 1) : t[0] / Math.max(m[0], 1);
  const nL = t[0] + dL * clamp(k, 0.3, 1.6);
  if (mC < 12) return polar(nL, tC, tH); // a near-grey body: colour it evenly instead of scaling its noise up
  return polar(nL, C * (tC / mC), tH + wrap(H - mH) * 0.35);
}

const GOLD = [[22, 12, 22], [48, 14, 45], [72, 6, 62], [92, -2, 48]];

// one look for one part; role 0 body, 1 feature, 2 white, 3 accent
function look(style, arg, role, L, C, H, lab, body, target) {
  switch (style) {
    case "tint":
      return role === 0 ? move(L, C, H, body, target) : lab;
    case "mono": {
      if (role === 1 || role === 2) return lab;
      const k = Math.hypot(target[1], target[2]) / Math.max(Math.hypot(body[1], body[2]), 1);
      return polar(L, C * k, Math.atan2(target[2], target[1]));
    }
    case "hue":
      return polar(L, C, H + (Number(arg || 0) * Math.PI) / 180);
    case "pastel":
      return polar(L * 0.55 + 45, C * 0.5, H);
    case "neon":
      return polar(L < 40 ? L * 0.6 : L, Math.min(C * 1.9, 128), H);
    case "gold": {
      const pos = clamp(L / 100, 0, 1) ** 0.9 * (GOLD.length - 1);
      const i = Math.min(GOLD.length - 2, Math.floor(pos)), f = pos - i;
      const g = GOLD[i].map((v, j) => v * (1 - f) + GOLD[i + 1][j] * f);
      return role === 1 ? [g[0] * 0.55, g[1] * 0.8, g[2] * 0.8] : g;
    }
    case "chrome":
      return [clamp(50 + (L - 50) * 1.35, 0, 100), -2, -6];
    case "clay":
      return [Math.round(L / 20) * 20 + 4, C * 0.55 * Math.cos(H) + 4, C * 0.55 * Math.sin(H) + 8];
    case "ink":
      return polar(L < 35 ? 18 : L < 70 ? 58 : 88, C * 1.2, H);
    case "negative":
      return polar(100 - L, C, H + Math.PI);
    default:
      return lab;
  }
}

// The body's mean colour, measured from the atlas itself: Lab averaged over pixels by body weight x alpha.
export function bodyLab(atlas, parts) {
  const d = atlas.data, p = parts.data;
  let s = 0, L = 0, A = 0, B = 0;
  for (let i = 0; i < d.length; i += 16) {          // every 4th pixel is plenty
    const w = (p[i] / 255) * (d[i + 3] / 255);
    if (w < 0.05) continue;
    const lab = toLab(d[i] / 255, d[i + 1] / 255, d[i + 2] / 255);
    s += w; L += w * lab[0]; A += w * lab[1]; B += w * lab[2];
  }
  return s ? [L / s, A / s, B / s] : [70, 0, 0];
}

export function recolor(atlas, parts, { style, arg }, body = null) {
  const out = new ImageData(new Uint8ClampedArray(atlas.data), atlas.width, atlas.height);
  if (!style) return out;
  const d = out.data, p = parts.data;
  body = body || bodyLab(atlas, parts);
  const target = style === "tint" || style === "mono" ? hexLab(arg) : null;
  for (let i = 0; i < d.length; i += 4) {
    if (d[i + 3] === 0) continue;
    const lab = toLab(d[i] / 255, d[i + 1] / 255, d[i + 2] / 255);
    const L = lab[0], C = Math.hypot(lab[1], lab[2]), H = Math.atan2(lab[2], lab[1]);
    const w = [p[i] / 255, p[i + 1] / 255, p[i + 2] / 255];
    w.push(Math.max(0, 1 - w[0] - w[1] - w[2]));
    let nl = 0, na = 0, nb = 0;
    for (let r = 0; r < 4; r++) {
      if (w[r] < 0.004) continue;
      const v = look(style, arg, r, L, C, H, lab, body, target);
      nl += w[r] * v[0]; na += w[r] * v[1]; nb += w[r] * v[2];
    }
    const sw = w[0] + w[1] + w[2] + w[3];
    const rgb = toRgb(nl / sw, na / sw, nb / sw);
    d[i] = rgb[0] * 255; d[i + 1] = rgb[1] * 255; d[i + 2] = rgb[2] * 255;
  }
  return out;
}
