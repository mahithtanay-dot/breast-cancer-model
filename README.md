# Breast Cancer Classifier

Fine-tunes a 3-billion-parameter LLM (Qwen2.5-3B) to classify breast tumors as benign or malignant, using the UCI Wisconsin Diagnostic Breast Cancer dataset.

> **⚠️ Not for clinical use.** This is a learning/research project — see [MODEL_CARD.md](MODEL_CARD.md) for training data limitations, evaluation gaps, and intended use before drawing any conclusions from its output.

## Disclaimer

This model **can be, and sometimes is, wrong.** It has not been clinically validated, is not a substitute for professional medical advice, diagnosis, or treatment, and must not be used as the basis for any real medical or health decision. Always consult a qualified healthcare provider.

This software is provided "as is," without warranty of any kind, express or implied. The author(s) accept no liability for any damages, losses, or legal claims arising from the use, misuse, or inability to use this software or its output. Use is entirely at your own risk.

## Data

`data/wdbc.data` — 569 patient samples, 30 numeric features per sample (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension — each as mean, standard error, and worst-case value), plus a diagnosis label (M = malignant, B = benign).

Source: [UCI Machine Learning Repository — Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic).

`data/seer_breast_cancer.csv` — 4,024 patients from the SEER (Surveillance, Epidemiology, and End Results) Program, 2006-2010, female patients with infiltrating duct and lobular carcinoma. Unlike `wdbc.data`, every patient here already has a confirmed cancer diagnosis — there's no benign class — so this dataset supports a **survival prediction task** (Alive vs. Dead from tumor stage, grade, size, hormone receptor status, and lymph node involvement), not the diagnosis task above. Not yet wired into a training notebook — reserved for a future prognosis-prediction pipeline.

## Approach

Each patient's 30 features are serialized into a text string (e.g. `mean_radius: 17.9900; mean_texture: 10.3800; ...`) and fed into Qwen2.5-3B with a classification head (`AutoModelForSequenceClassification`, 2 labels) attached. The base model loads in 4-bit (QLoRA) and stays frozen; only LoRA adapters plus the classification head are trained, which keeps this within Google Colab's free-tier T4 GPU (~15GB VRAM).

**Worth noting:** a 3B-parameter LLM is a lot of firepower for 30 numeric features and 569 rows — a small MLP or gradient-boosted tree would likely match or beat it with a fraction of the compute (see `BreastCancer.ipynb` for a simple logistic regression baseline on the same data). This project's goal was learning the LLM fine-tuning pipeline (QLoRA, PEFT, Colab free-tier constraints) rather than picking the best tool for this specific dataset.

## Repo contents

| File | Purpose |
|---|---|
| `finetune_3B_lora.ipynb` | Main pipeline: load data, QLoRA fine-tune Qwen2.5-3B, evaluate, save adapter to Drive, run predictions on new patients |
| `BreastCancer.ipynb` | Simple logistic regression baseline for comparison |
| `data/wdbc.data` | Raw dataset |
| `data/breast+cancer+wisconsin+diagnostic.zip` | Original UCI download |

## Running it

Open directly in Colab:

👉 https://colab.research.google.com/github/mahithtanay-dot/breast-cancer-model/blob/master/finetune_3B_lora.ipynb

1. **Runtime → Change runtime type → T4 GPU**
2. **Runtime → Run all**
3. Approve the Google Drive mount prompt when it appears (checkpoints and the final adapter save there, so they survive a disconnect)

Training takes roughly 10-30 minutes on a free-tier T4. Results (accuracy, F1, AUC, confusion matrix) print at the end of Section 8.

## Making predictions

Section 10 exposes a `predict_patient(measurements)` function — pass a dict with all 30 feature values and get back a prediction, confidence, and per-class probabilities. Works in a fresh Colab session too, since it reloads the saved adapter from Drive rather than requiring the training run to still be in memory.
