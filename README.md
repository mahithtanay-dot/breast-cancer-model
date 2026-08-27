# Breast Cancer Classifier

Two models answering the same question — is this cancerous? — from two different kinds of data:

1. **Tabular model** — a fine-tuned 3-billion-parameter LLM (Qwen2.5-3B + LoRA) that classifies a tumor as benign or malignant from numeric cytology measurements.
2. **Image model** — a small CNN trained from scratch that classifies a breast tissue microscope image patch as cancerous or not.

> **⚠️ Not for clinical use.** This is a learning/research project — see [MODEL_CARD.md](MODEL_CARD.md) for training data limitations, evaluation gaps, and intended use before drawing any conclusions from its output.

## Disclaimer

This model **can be, and sometimes is, wrong.** It has not been clinically validated, is not a substitute for professional medical advice, diagnosis, or treatment, and must not be used as the basis for any real medical or health decision. Always consult a qualified healthcare provider.

This software is provided "as is," without warranty of any kind, express or implied. The author(s) accept no liability for any damages, losses, or legal claims arising from the use, misuse, or inability to use this software or its output. Use is entirely at your own risk.

## Try it

A live demo is hosted on Hugging Face Spaces — upload a tissue image or enter numeric measurements and get an instant prediction. *(Link goes here once deployed.)*

The demo runs the two trained models directly; no setup needed to use it.

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
| `webapp/app.py` | Gradio app serving both trained models for the live demo |
| `webapp/requirements.txt` | Dependencies for the demo app |

## Running the demo yourself

`webapp/app.py` loads both trained models from Hugging Face Hub and serves them through a two-tab Gradio interface (image upload, or numeric measurement entry). To run your own copy:

1. `pip install -r webapp/requirements.txt`
2. `python webapp/app.py`

Note: the numeric-measurements tab loads a 3B-parameter LLM in full precision (4-bit quantization requires a GPU, so it won't apply on CPU-only hardware), so it's slow — 30-60+ seconds per prediction on CPU. The image tab runs a tiny CNN and is fast anywhere.
