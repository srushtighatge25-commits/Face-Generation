from __future__ import annotations

import io
import os
import secrets
import threading
from pathlib import Path

import torch
from flask import Flask, jsonify, render_template, request, send_file
from PIL import Image

from models import Generator, load_generator


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.environ.get("FACE_MODEL_PATH", "generator.pth"))
if not MODEL_PATH.is_absolute():
    MODEL_PATH = BASE_DIR / MODEL_PATH
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = Flask(__name__)
inference_lock = threading.Lock()



generator = load_generator(MODEL_PATH, DEVICE)


def tensor_to_png(image_tensor: torch.Tensor) -> io.BytesIO:
    """Convert one output tensor in [-1, 1] to an in-memory PNG."""
    image = image_tensor.detach().cpu().squeeze(0)
    image = image.clamp(-1, 1).add(1).div(2).mul(255).byte()
    image = image.permute(1, 2, 0).contiguous()
    height, width, channels = image.shape
    if channels != 3:
        raise ValueError(f"Expected an RGB image, but the model produced {channels} channels.")

    # Convert directly to bytes instead of calling Tensor.numpy(). This keeps
    # inference working in lightweight environments where NumPy is not installed.
    pixel_bytes = bytes(image.view(-1).tolist())

    buffer = io.BytesIO()
    Image.frombytes("RGB", (width, height), pixel_bytes).save(
        buffer, format="PNG", optimize=True
    )
    buffer.seek(0)
    return buffer


@app.get("/")
def index():
    return render_template(
        "index.html",
        device="GPU" if DEVICE.type == "cuda" else "CPU",
        image_size=64,
        latent_size=generator.latent_size,
    )


@app.post("/generate")
def generate_face():
    seed_text = request.args.get("seed")
    if seed_text is not None and (
        not seed_text.isascii() or not seed_text.isdecimal() or len(seed_text) > 10
        or int(seed_text) > 4294967295
    ):
        return jsonify(error="Seed must be a whole number from 0 to 4294967295."), 400
    seed = int(seed_text) if seed_text is not None else secrets.randbits(32)
    try:
        with inference_lock, torch.inference_mode():
            random = torch.Generator(device=DEVICE).manual_seed(seed)
            noise = torch.randn(1, generator.latent_size, 1, 1, device=DEVICE, generator=random)
            generated_image = generator(noise)
            png = tensor_to_png(generated_image)

        response = send_file(
            png,
            mimetype="image/png",
            download_name=f"face-{seed}.png",
        )
        response.headers["Cache-Control"] = "no-store, max-age=0"
        response.headers["X-Generation-Seed"] = str(seed)
        return response
    except Exception:
        app.logger.exception("Face generation failed")
        return jsonify(
            error="Face generation failed. Please try again.",
        ), 500


@app.get("/health")
def health():
    return jsonify(status="ok", device=DEVICE.type, model=MODEL_PATH.name)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
