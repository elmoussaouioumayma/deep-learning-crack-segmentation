"""Gradio demo: upload a photo of a concrete or asphalt surface, get the crack mask.

    python app.py            # local
Also runs unchanged as a Hugging Face Space (SDK: gradio, app file: app.py).
"""

import gradio as gr
import numpy as np

from predict import load_model, overlay, predict_probs

MODEL, THRESHOLD = load_model("weights/unet.pt")


def run(image, threshold):
    if image is None:
        return None, None, ""
    rgb = np.asarray(image.convert("RGB"))
    prob = predict_probs(MODEL, rgb)
    mask = prob > threshold
    heat = (np.stack([prob] * 3, -1) * 255).astype(np.uint8)
    info = f"Crack pixels: {100 * mask.mean():.2f} % of the image  |  threshold {threshold:.2f}"
    return overlay(rgb, mask), heat, info


with gr.Blocks(title="Crack segmentation") as demo:
    gr.Markdown("## Crack segmentation with a U-Net\nUpload a photo of a concrete or asphalt surface. "
                "The model marks every pixel that belongs to a crack.")
    with gr.Row():
        inp = gr.Image(type="pil", label="Photo")
        out = gr.Image(label="Detected cracks")
        heat = gr.Image(label="Crack probability")
    thr = gr.Slider(0.05, 0.98, value=THRESHOLD, step=0.01, label="Threshold (default: chosen on validation data)")
    info = gr.Markdown()
    gr.Button("Detect cracks", variant="primary").click(run, [inp, thr], [out, heat, info])

if __name__ == "__main__":
    demo.launch()
