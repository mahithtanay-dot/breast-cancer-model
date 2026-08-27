# Breast Cancer Classifier

Two models answering the same question — is this cancerous? — from two different kinds of data:

1. **Tabular model** — a fine-tuned 3-billion-parameter LLM (Qwen2.5-3B + LoRA) that classifies a tumor as benign or malignant from numeric cytology measurements.
2. **Image model** — a small CNN trained from scratch that classifies a breast tissue microscope image patch as cancerous or not.

> **⚠️ Not for clinical use.** This is a learning/research project — see [MODEL_CARD.md](MODEL_CARD.md) for training data limitations, evaluation gaps, and intended use before drawing any conclusions from its output.

## Disclaimer

This model **can be, and sometimes is, wrong.** It has not been clinically validated, is not a substitute for professional medical advice, diagnosis, or treatment, and must not be used as the basis for any real medical or health decision. Always consult a qualified healthcare provider.

This software is provided "as is," without warranty of any kind, express or implied. The author(s) accept no liability for any damages, losses, or legal claims arising from the use, misuse, or inability to use this software or its output. Use is entirely at your own risk.

## Try it

**Image model (always-on):** [huggingface.co/spaces/mahithtanay/breast-cancer-classifier-demo](https://huggingface.co/spaces/mahithtanay/breast-cancer-classifier-demo) — upload a tissue image patch and get an instant prediction. Runs entirely in your browser (no server, no image ever uploaded anywhere) via ONNX.

**Numeric model (on-demand):** the 3B-parameter LLM is too large for free hosting, so instead of a permanent link, it's launched on demand as a temporary public link from the training notebook's Gradio cell. Ask the maintainer for a current link, or see "Running the demo yourself" below.

## Approach

**Tabular model:** cytology measurements are serialized into a text string (e.g. `mean_radius: 17.9900; mean_texture: 10.3800; ...`) and fed into Qwen2.5-3B with a classification head attached. The base model is frozen and quantized to 4-bit (QLoRA); only LoRA adapters plus the classification head are trained. Worth noting: a 3B-parameter LLM is a lot of firepower for a few dozen numbers — a small MLP or gradient-boosted tree would likely match or beat it with a fraction of the compute. This project's goal was learning the fine-tuning pipeline (QLoRA, PEFT, working within free-tier GPU constraints), not picking the best tool for this specific dataset.

**Image model:** follows the same right-sized-model philosophy — rather than fine-tuning a huge pretrained vision model on 50x50 pixel patches, it trains a small custom CNN (~100K parameters, three conv blocks) from scratch, a much better match for images this small and this task.

Training data: ~1,252 samples (two combined UCI Wisconsin datasets) for the tabular model, 25,000 sampled real histopathology patches (from a pool of 277,524) for the image model. See [MODEL_CARD.md](MODEL_CARD.md) for full details, sources, and limitations (small sample sizes, single-institution data, no external validation).

## About the training code

The notebooks, raw datasets, and training pipeline used to produce these models are kept in a private companion repository rather than published here. This repo is intended for people to **use** the trained models (via the live demo) and understand **how and why** they were built (via this README and the model card) — not to hand out a one-click retraining pipeline.

## Repo contents

| File | Purpose |
|---|---|
| `MODEL_CARD.md` | Full model documentation: training data, evaluation, limitations, intended use |
| `webapp/static-site/index.html` | The deployed image-classifier demo — a self-contained static page that runs the CNN client-side via ONNX (`onnxruntime-web`) |

## Running the demo yourself

The image demo is a single static HTML file with no build step or server — open `webapp/static-site/index.html` directly in a browser, or deploy it anywhere that serves static files (it fetches the model weights directly from the public Hugging Face Hub repo at runtime).

The numeric model can't run this way (it needs a GPU + several GB of RAM), so its demo is instead launched on demand from inside the training notebook (kept in the private companion repo) using Gradio's `share=True`, which prints a temporary public link.
