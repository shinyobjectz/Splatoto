// <splatoto-emoji> — a Splatoto emoji that turns toward the cursor (or spins), drawn from a sprite atlas.
//
//   <script type="module" src="splatoto-emoji.js"></script>
//   <splatoto-emoji code="1f600" base="./sprites/" size="64"></splatoto-emoji>
//   <splatoto-emoji code="1f680" mode="spin"></splatoto-emoji>
//
// Attributes:
//   code   Noto's file code: lowercase hex code points joined by "_", no fe0f (😀 = 1f600, ❤️ = 2764)
//   base   folder holding <code>.webp (cone) and <code>_360.webp (full turn); default "./sprites/"
//   size   CSS pixels; default 64
//   mode   "follow" (default): turn toward the cursor inside the cone (yaw ±45°, tilt ±25°)
//          "spin": turn all the way round (needs <code>_360.webp)
//          "still": the front view only
//   reach  CSS pixels from the emoji's centre at which it is fully turned; default 480
//   recolor  a look from recolor.js, e.g. "tint #3a7bff", "mono #ff4fa8", "hue 180", "gold", "chrome", "pastel",
//          "neon", "clay", "ink", "negative". Needs the parts atlas beside the picture (<code>_parts.webp or
//          <code>_360_parts.webp) and images the page may read (same origin, or served with CORS)
//
// Atlas layout (see README): rows are camera pitch +25, +12.5, 0, −12.5, −25 (camera above first); columns are yaw.
// cone: 9 columns, yaw −45..45 by 11.25. 360: 32 columns, yaw 0..348.75 by 11.25. Positive yaw shows the emoji turned
// toward the picture's left, so a cursor on the right picks a column left of centre.

import { recolor, parseLook } from "./recolor.js";

const ROWS = 5;
const cache = new Map(); // "url|look" -> object URL of the recoloured atlas

async function pixels(url) {
  const res = await fetch(url, { mode: "cors" });
  if (!res.ok) throw new Error(`${url}: ${res.status}`);
  const bmp = await createImageBitmap(await res.blob());
  const c = new OffscreenCanvas(bmp.width, bmp.height);
  const g = c.getContext("2d");
  g.drawImage(bmp, 0, 0);
  return g.getImageData(0, 0, bmp.width, bmp.height);
}

async function recoloured(url, partsUrl, lookText) {
  const key = `${url}|${lookText}`;
  if (!cache.has(key)) {
    cache.set(key, (async () => {
      const [atlas, parts] = await Promise.all([pixels(url), pixels(partsUrl)]);
      const out = recolor(atlas, parts, parseLook(lookText));
      const c = new OffscreenCanvas(out.width, out.height);
      c.getContext("2d").putImageData(out, 0, 0);
      return URL.createObjectURL(await c.convertToBlob({ type: "image/png" }));
    })());
  }
  return cache.get(key);
}
const CONE_COLS = 9;
const TURN_COLS = 32;

class SplatotoEmoji extends HTMLElement {
  static observedAttributes = ["code", "base", "size", "mode", "recolor"];

  constructor() {
    super();
    this.attachShadow({ mode: "open" });
    this.shadowRoot.innerHTML = `<style>
      :host { display: inline-block; width: var(--s); height: var(--s); vertical-align: middle; }
      div { width: 100%; height: 100%; background-repeat: no-repeat; image-rendering: auto; }
    </style><div part="sprite" role="img"></div>`;
    this.el = this.shadowRoot.querySelector("div");
    this.cur = { x: 0, y: 0 };
    this.target = { x: 0, y: 0 };
    this.onMove = (e) => this.aim(e.clientX, e.clientY);
    this.frame = this.frame.bind(this);
  }

  connectedCallback() {
    this.render();
    addEventListener("pointermove", this.onMove, { passive: true });
    this.raf = requestAnimationFrame(this.frame);
  }

  disconnectedCallback() {
    removeEventListener("pointermove", this.onMove);
    cancelAnimationFrame(this.raf);
  }

  attributeChangedCallback() {
    if (this.isConnected) this.render();
  }

  get mode() {
    return this.getAttribute("mode") || "follow";
  }

  render() {
    const code = this.getAttribute("code") || "1f600";
    const base = this.getAttribute("base") || "./sprites/";
    const size = Number(this.getAttribute("size") || 64);
    const spin = this.mode === "spin";
    this.cols = spin ? TURN_COLS : CONE_COLS;
    this.style.setProperty("--s", `${size}px`);
    const stem = `${base}${code}${spin ? "_360" : ""}`;
    this.el.style.backgroundImage = `url("${stem}.webp")`;
    const look = this.getAttribute("recolor");
    this.want = look ? `${stem}|${look}` : null; // a late result for an older look is dropped
    if (look) {
      const want = this.want;
      recoloured(`${stem}.webp`, `${stem}_parts.webp`, look)
        .then((u) => { if (this.want === want) this.el.style.backgroundImage = `url("${u}")`; })
        .catch((e) => console.warn("splatoto recolor:", e));
    }
    this.el.style.backgroundSize = `${this.cols * 100}% ${ROWS * 100}%`;
    this.el.setAttribute("aria-label", this.getAttribute("label") || code);
    this.show(spin ? 0 : 4, 2);
  }

  aim(px, py) {
    if (this.mode !== "follow") return;
    const r = this.getBoundingClientRect();
    const reach = Number(this.getAttribute("reach") || 480);
    const clamp = (v) => Math.max(-1, Math.min(1, v));
    this.target.x = clamp((px - (r.left + r.width / 2)) / reach);
    this.target.y = clamp((py - (r.top + r.height / 2)) / reach);
  }

  show(col, row) {
    const x = (col / (this.cols - 1)) * 100;
    const y = (row / (ROWS - 1)) * 100;
    this.el.style.backgroundPosition = `${x}% ${y}%`;
  }

  frame(t) {
    this.raf = requestAnimationFrame(this.frame);
    if (this.mode === "spin") {
      this.show(Math.floor((t / 1000) * 8) % TURN_COLS, 2); // 8 views a second: one turn in 4 s
      return;
    }
    if (this.mode !== "follow") return;
    this.cur.x += (this.target.x - this.cur.x) * 0.18;
    this.cur.y += (this.target.y - this.cur.y) * 0.18;
    // cursor right -> negative yaw (left columns); cursor above -> negative pitch (lower rows)
    this.show(Math.round(4 - this.cur.x * 4), Math.round(2 - this.cur.y * 2));
  }
}

customElements.define("splatoto-emoji", SplatotoEmoji);
