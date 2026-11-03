"""Gradio demo: upload a photo of a concrete or asphalt surface, get the crack mask.

    python app.py            # local
Also runs unchanged as a Hugging Face Space (SDK: gradio, app file: app.py).
"""

import json
from pathlib import Path

import gradio as gr
import numpy as np

from classical import ClassicalParams, segment
from predict import load_model, overlay, predict_probs

MODEL, THRESHOLD = load_model("weights/unet.pt")
_saved = json.loads(Path("results/classical_params.json").read_text())
CLASSICAL = ClassicalParams(**{k: _saved[k] for k in ("kernel", "sigma", "k", "min_area", "min_elong")})
MAX_SIDE = 1600


def run(image, threshold):
    if image is None:
        raise gr.Error("Upload a photo first.")
    image = image.convert("RGB")
    if max(image.size) > MAX_SIDE:                      # keep the demo fast on a free CPU
        image.thumbnail((MAX_SIDE, MAX_SIDE))
    rgb = np.asarray(image)
    prob = predict_probs(MODEL, rgb)
    mask = prob > threshold
    classic = segment(rgb, CLASSICAL)
    heat = (np.stack([prob] * 3, -1) * 255).astype(np.uint8)
    info = (f"**U-Net:** {100 * mask.mean():.2f} % of the pixels are crack (threshold {threshold:.2f})  ·  "
            f"**classical baseline:** {100 * classic.mean():.2f} %")
    return overlay(rgb, mask), heat, overlay(rgb, classic, color=(40, 120, 255)), info


with gr.Blocks(title="Crack segmentation") as demo:
    gr.Markdown("## Crack segmentation: U-Net vs. classical image processing\n"
                "Upload a photo of a concrete or asphalt surface. The U-Net marks every pixel that belongs to a "
                "crack; the classical pipeline (black-hat filter + threshold) is shown for comparison.")
    with gr.Row():
        inp = gr.Image(type="pil", label="Photo")
        out = gr.Image(label="U-Net: detected cracks")
    with gr.Row():
        heat = gr.Image(label="U-Net: crack probability")
        classic = gr.Image(label="Classical baseline")
    thr = gr.Slider(0.05, 0.98, value=round(THRESHOLD, 2), step=0.01,
                    label="U-Net threshold (default: chosen on validation images)")
    info = gr.Markdown()
    btn = gr.Button("Detect cracks", variant="primary")
    btn.click(run, [inp, thr], [out, heat, classic, info])
    inp.upload(run, [inp, thr], [out, heat, classic, info])

if __name__ == "__main__":
    demo.launch()
