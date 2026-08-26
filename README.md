# Breast Cancer Classifier

Fine-tunes a 3-billion-parameter LLM (Qwen2.5-3B) to classify breast tumors as benign or malignant, using the UCI Wisconsin Diagnostic Breast Cancer dataset.

> **⚠️ Not for clinical use.** This is a learning/research project — see [MODEL_CARD.md](MODEL_CARD.md) for training data limitations, evaluation gaps, and intended use before drawing any conclusions from its output.

## Disclaimer

This model **can be, and sometimes is, wrong.** It has not been clinically validated, is not a substitute for professional medical advice, diagnosis, or treatment, and must not be used as the basis for any real medical or health decision. Always consult a qualified healthcare provider.

This software is provided "as is," without warranty of any kind, express or implied. The author(s) accept no liability for any damages, losses, or legal claims arising from the use, misuse, or inability to use this software or its output. Use is entirely at your own risk.

## Data

Two same-task datasets are combined into one training pool of ~1,252 samples (after dropping 16 rows with missing values):

- `data/wdbc.data` — 569 patient samples, 30 numeric features per sample (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension — each as mean, standard error, and worst-case value), plus a diagnosis label (M = malignant, B = benign). Source: [UCI — Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic).
- `data/wisconsin_original.data` — 699 patient samples, a *different* set of 9 features (clump thickness, uniformity of cell size/shape, marginal adhesion, single epithelial cell size, bare nuclei, bland chromatin, normal nucleoli, mitoses), same benign/malignant task. Source: [UCI — Breast Cancer Wisconsin (Original)](https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original).

These two don't share a feature schema, but since rows are serialized to text before being fed to the model (rather than a fixed-width numeric vector), both can be mixed into the same training set as long as they share the same label space — the model just reads whichever key:value pairs are present per row. Both are still from the same institution/era, so this adds volume and some feature diversity, but does **not** provide true external validation (see [MODEL_CARD.md](MODEL_CARD.md)).

`data/seer_breast_cancer.csv` — 4,024 patients from the SEER (Surveillance, Epidemiology, and End Results) Program, 2006-2010, female patients with infiltrating duct and lobular carcinoma. Unlike `wdbc.data`, every patient here already has a confirmed cancer diagnosis — there's no benign class — so this dataset supports a **survival prediction task** (Alive vs. Dead from tumor stage, grade, size, hormone receptor status, and lymph node involvement), not the diagnosis task above. Not yet wired into a training notebook — reserved for a future prognosis-prediction pipeline.

## Approach

Each patient's 30 features are serialized into a text string (e.g. `mean_radius: 17.9900; mean_texture: 10.3800; ...`) and fed into Qwen2.5-3B with a classification head (`AutoModelForSequenceClassification`, 2 labels) attached. The base model loads in 4-bit (QLoRA) and stays frozen; only LoRA adapters plus the classification head are trained, which keeps this within Google Colab's free-tier T4 GPU (~15GB VRAM).

**Worth noting:** a 3B-parameter LLM is a lot of firepower for 30 numeric features and 569 rows — a small MLP or gradient-boosted tree would likely match or beat it with a fraction of the compute (see `BreastCancer.ipynb` for a simple logistic regression baseline on the same data). This project's goal was learning the LLM fine-tuning pipeline (QLoRA, PEFT, Colab free-tier constraints) rather than picking the best tool for this specific dataset.

## Repo contents

| File | Purpose |
|---|---|
| `finetune_3B_lora.ipynb` | Main pipeline: load data, QLoRA fine-tune Qwen2.5-3B, evaluate, save adapter to Drive, run predictions on new patients |
| `BreastCancer.ipynb` | Simple logistic regression baseline for comparison |
| `data/wdbc.data` | UCI Wisconsin (Diagnostic) raw dataset |
| `data/wisconsin_original.data` | UCI Wisconsin (Original) raw dataset |
| `data/breast+cancer+wisconsin+diagnostic.zip` | Original UCI download |
| `data/seer_breast_cancer.csv` | SEER survival dataset, reserved for a future prognosis-prediction pipeline (different task, see Data section) |

## Running it

Open directly in Colab:

👉 https://colab.research.google.com/github/mahithtanay-dot/breast-cancer-model/blob/master/finetune_3B_lora.ipynb

1. **Runtime → Change runtime type → T4 GPU**
2. **Runtime → Run all**
3. Approve the Google Drive mount prompt when it appears (checkpoints and the final adapter save there, so they survive a disconnect)

Training takes roughly 10-30 minutes on a free-tier T4. Results (accuracy, F1, AUC, confusion matrix) print at the end of Section 8.

## Making predictions

Section 10 exposes a `predict_patient(measurements)` function — pass a dict with all 30 feature values and get back a prediction, confidence, and per-class probabilities. Works in a fresh Colab session too, since it reloads the saved adapter from Drive rather than requiring the training run to still be in memory.
