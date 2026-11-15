"""Noto 3D emoji -> Idol (the PyTorch release, CUDA) on Modal.

    modal run emoji_modal.py --codes 1f600,1f436 [--gpu L40S]
    modal run emoji_modal.py --list codes.txt [--gpu L40S] [--containers 10]

Reads /vol/emoji/in/CODE.png (Noto 3D, 512 px) on the volume idol-work; writes /vol/emoji/out/CODE.ply (3DGS, 262,144
Gaussians), CODE_camera.json (Idol's predicted input camera) and CODE_input.png (what the model saw). Skips a code
whose .ply exists. The two image encoders are the ungated sources proved on 11-10 (tools/idol/steps/common.py):
timm's DINOv3 ViT-H+/16 and FLUX.2-klein-4B's VAE, both already in the volume's Hugging Face cache, so no token.
"""
import json, os, time
import modal

SDK = os.environ.get("IDOL_SDK", os.path.expanduser("~/idol-sdk"))   # a checkout of ivanfioravanti/idol-sdk-mlx @ ad3a9e3
vol = modal.Volume.from_name("idol-work")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("torch==2.8.0", "torchvision==0.23.0", "transformers>=4.56", "diffusers>=0.40", "accelerate>=1.0",
                    "safetensors>=0.4", "tensordict>=0.8", "numpy>=1.26", "pillow>=10.0", "plyfile>=1.0", "tqdm",
                    "timm>=1.0.20", "huggingface_hub>=0.34")
    .env({"HF_HOME": "/vol/hf", "HF_HUB_OFFLINE": "1", "PYTHONUNBUFFERED": "1"})
    .add_local_dir(SDK, "/sdk", ignore=[".git", "**/__pycache__", "assets", "demo", "perf", "tests"])
)
app = modal.App("raea-emoji3d", image=image)


def prep(src, dst):
    """The same input as the local runs: the emoji cropped to its pixels, 820 px on its long side, on mid grey."""
    from PIL import Image
    im = Image.open(src).convert("RGBA"); im = im.crop(im.getbbox())
    k = 820 / max(im.size); im = im.resize((round(im.width * k), round(im.height * k)), Image.LANCZOS)
    bg = Image.new("RGB", (1024, 1024), (127, 127, 127)); bg.paste(im, ((1024 - im.width) // 2, (1024 - im.height) // 2), im)
    bg.save(dst)


@app.cls(gpu="L40S", volumes={"/vol": vol}, timeout=3600, memory=32768, max_containers=10)
class Idol:
    @modal.enter()
    def load(self):
        import sys
        sys.path.insert(0, "/sdk")
        import torch, timm
        from huggingface_hub import hf_hub_download, snapshot_download
        from safetensors.torch import load_file
        from diffusers.models import AutoencoderKLFlux2
        import idol.pipeline as P
        from idol.conditioning import DTYPES

        class TimmTokens(torch.nn.Module):
            """timm's DINOv3 answering like transformers' DINOv3ViTModel: .last_hidden_state [B, 1 + 4 + patches, 1280]."""
            def __init__(self, m):
                super().__init__(); self.m = m

            def forward(self, pixel_values):
                from types import SimpleNamespace
                return SimpleNamespace(last_hidden_state=self.m.forward_features(pixel_values))

        class Ungated(P.ImageEncoder):
            def __init__(self, dinov3_repo, flux2_vae_repo, dinov3_dtype="bf16", device="cuda"):
                self.device = torch.device(device)
                m = timm.create_model("vit_huge_plus_patch16_dinov3.lvd1689m", pretrained=False, num_classes=0)
                m.load_state_dict(load_file(hf_hub_download("timm/vit_huge_plus_patch16_dinov3.lvd1689m", "model.safetensors")), strict=True)
                self.dinov3 = TimmTokens(m).to(DTYPES[dinov3_dtype]).eval().requires_grad_(False).to(self.device)
                self.flux2_vae = AutoencoderKLFlux2.from_pretrained("black-forest-labs/FLUX.2-klein-4B", subfolder="vae",
                                                                    torch_dtype=torch.bfloat16).eval().requires_grad_(False).to(self.device)

        P.require_encoder_access = lambda *a, **k: None
        P.ImageEncoder = Ungated
        t0 = time.time()
        self.pipe = P.IdolPipeline(snapshot_download("speridlabs/idol", allow_patterns=["*.json", "*.safetensors"]), device="cuda")
        self.load_s = time.time() - t0
        self.n = 0

    @modal.method()
    def run(self, code: str):
        out_dir = "/vol/emoji/out"
        os.makedirs(out_dir, exist_ok=True)
        if os.path.exists(f"{out_dir}/{code}.ply"):
            return {"code": code, "skipped": True}
        prep(f"/vol/emoji/in/{code}.png", "/tmp/in.png")
        t0 = time.time()
        o = self.pipe("/tmp/in.png", mode="synthetic", seed=42, num_gaussians=262144)
        o.gaussian.save_ply(f"{out_dir}/{code}.ply.part"); os.replace(f"{out_dir}/{code}.ply.part", f"{out_dir}/{code}.ply")
        json.dump(o.camera, open(f"{out_dir}/{code}_camera.json", "w"))
        o.preprocessed_image.save(f"{out_dir}/{code}_input.png")
        self.n += 1
        if self.n % 10 == 1:
            vol.commit()
        return {"code": code, "seconds": round(time.time() - t0, 2), "timings": {k: round(v, 2) for k, v in o.timings.items()},
                "camera": o.camera, "load_s": round(self.load_s, 1), "nth_in_container": self.n}

    @modal.exit()
    def flush(self):
        vol.commit()


fetch_image = modal.Image.debian_slim(python_version="3.11").apt_install("git")
NOTO_COMMIT = "99701e0"   # googlefonts/noto-emoji main on 2026-10-26, the checkout this pipeline was tried on


@app.function(image=fetch_image, volumes={"/vol": vol}, timeout=1800)
def fetch() -> list:
    """Noto's 3D/png/512 (the whole set) into /vol/emoji/in as CODE.png, at the pinned commit; returns the codes."""
    import shutil, subprocess
    work = "/tmp/noto"
    subprocess.run(["git", "clone", "-q", "--filter=blob:none", "--sparse", "https://github.com/googlefonts/noto-emoji.git", work], check=True)
    subprocess.run(["git", "-C", work, "checkout", "-q", NOTO_COMMIT], check=True)
    subprocess.run(["git", "-C", work, "sparse-checkout", "set", "3D/png/512"], check=True)
    os.makedirs("/vol/emoji/in", exist_ok=True)
    codes = []
    for name in sorted(os.listdir(f"{work}/3D/png/512")):
        if name.startswith("emoji_u") and name.endswith(".png"):
            code = name[len("emoji_u"):-len(".png")]
            shutil.copyfile(f"{work}/3D/png/512/{name}", f"/vol/emoji/in/{code}.png")
            codes.append(code)
    vol.commit()
    return codes


@app.local_entrypoint()
def main(codes: str = "", list: str = "", gpu: str = "L40S", containers: int = 10, fetch_all: bool = False):
    if fetch_all:
        got = fetch.remote()
        open("all_codes.txt", "w").write("\n".join(got) + "\n")
        print(f"== fetched {len(got)} codes -> all_codes.txt")
        return
    todo = [c for c in codes.split(",") if c] or [l.strip() for l in open(list) if l.strip()]
    t0 = time.time()
    cls = Idol.with_options(gpu=gpu, max_containers=containers)
    with open(f"runs_{gpu}.jsonl", "a") as log:
        for r in cls().run.map(todo, return_exceptions=True):
            r = r if isinstance(r, dict) else {"error": repr(r)[:300]}
            log.write(json.dumps(r) + "\n"); log.flush()
            print(json.dumps(r)[:240])
    print(f"== {len(todo)} in {time.time() - t0:.0f}s on {gpu}")
