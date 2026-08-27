# Breast Cancer Classifier

Two classifiers for the same underlying question — is this cancerous? — built on two different kinds of data:

1. **`finetune_3B_lora.ipynb`** — fine-tunes Qwen2.5-3B (an LLM) on ~1,252 rows of hand-engineered numeric cytology features (UCI Wisconsin datasets)
2. **`idc_histopathology_classifier.ipynb`** — trains a small CNN from scratch on 25,000 real breast tissue microscope image patches (IDC dataset)

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

**IDC histopathology images** (used by `idc_histopathology_classifier.ipynb`, downloaded directly inside that notebook, not stored in this repo — it's ~1.6GB): 277,524 real 50x50 pixel image patches extracted from 162 patients' breast tissue whole-slide biopsy images, each labeled cancerous (IDC-positive) or non-cancerous. The notebook samples a balanced 25,000-patch subset (12,500 per class; the full set is naturally imbalanced, ~28% positive) for training. Source: [Janowczyk & Madabhushi, IDC_regular_ps50_idx5](https://andrewjanowczyk.com/use-case-6-invasive-ductal-carcinoma-idc-segmentation/). Unlike the datasets above, these are genuinely large-scale and this is real microscope imagery rather than pre-computed numeric summaries — but it's still all from one dataset/collection process, so the same external-validation caveat applies.

## Approach

Each patient's 30 features are serialized into a text string (e.g. `mean_radius: 17.9900; mean_texture: 10.3800; ...`) and fed into Qwen2.5-3B with a classification head (`AutoModelForSequenceClassification`, 2 labels) attached. The base model loads in 4-bit (QLoRA) and stays frozen; only LoRA adapters plus the classification head are trained, which keeps this within Google Colab's free-tier T4 GPU (~15GB VRAM).

**Worth noting:** a 3B-parameter LLM is a lot of firepower for 30 numeric features and 569 rows — a small MLP or gradient-boosted tree would likely match or beat it with a fraction of the compute (see `BreastCancer.ipynb` for a simple logistic regression baseline on the same data). This project's goal was learning the LLM fine-tuning pipeline (QLoRA, PEFT, Colab free-tier constraints) rather than picking the best tool for this specific dataset.

`idc_histopathology_classifier.ipynb` follows the same right-sized-model philosophy on the image side: rather than fine-tuning a huge pretrained vision model on 50x50 pixel patches, it trains a small custom CNN (~100K parameters — three conv blocks) from scratch, which is a much better match for images this small and this task.

## Repo contents

| File | Purpose |
|---|---|
| `finetune_3B_lora.ipynb` | Tabular pipeline: load data, QLoRA fine-tune Qwen2.5-3B, evaluate, save adapter to Drive, run predictions on new patients |
| `idc_histopathology_classifier.ipynb` | Image pipeline: download IDC histopathology patches, train a small CNN from scratch, evaluate, predict on new images |
| `BreastCancer.ipynb` | Simple logistic regression baseline for comparison |
| `data/wdbc.data` | UCI Wisconsin (Diagnostic) raw dataset |
| `data/wisconsin_original.data` | UCI Wisconsin (Original) raw dataset |
| `data/breast+cancer+wisconsin+diagnostic.zip` | Original UCI download |
| `data/seer_breast_cancer.csv` | SEER survival dataset, reserved for a future prognosis-prediction pipeline (different task, see Data section) |

## Running it

Open either notebook directly in Colab:

👉 [`finetune_3B_lora.ipynb`](https://colab.research.google.com/github/mahithtanay-dot/breast-cancer-model/blob/master/finetune_3B_lora.ipynb) (tabular, LLM fine-tuning)

👉 [`idc_histopathology_classifier.ipynb`](https://colab.research.google.com/github/mahithtanay-dot/breast-cancer-model/blob/master/idc_histopathology_classifier.ipynb) (images, CNN from scratch)

For either:
1. **Runtime → Change runtime type → T4 GPU**
2. **Runtime → Run all**
3. Approve the Google Drive mount prompt when it appears (checkpoints and the final model save there, so they survive a disconnect)

The tabular notebook takes roughly 10-30 minutes on a free-tier T4; results (accuracy, F1, AUC, confusion matrix) print at the end of Section 8. The image notebook additionally spends a few minutes downloading and extracting the 1.6GB dataset before training (Section 1), with results printing at the end of Section 8 there too.

## Making predictions

- `finetune_3B_lora.ipynb` Section 10 exposes a `predict_patient(measurements)` function — pass a dict with all 30 feature values and get back a prediction, confidence, and per-class probabilities. Works in a fresh Colab session too, since it reloads the saved adapter from Drive.
- `idc_histopathology_classifier.ipynb` Section 9 exposes a `predict_patch(image_path)` function — pass a path to a 50x50 tissue patch image and get back the same kind of prediction/confidence output.

## Live demo (Hugging Face Spaces)

`webapp/app.py` is a Gradio app with two tabs — upload a tissue image (fast, runs the CNN) or enter numeric measurements (slow, runs the 3B LLM on CPU — see note below). To deploy your own copy:

1. **Train both models in Colab** (both notebooks above), then run each notebook's final "Publish to Hugging Face Hub" section — this needs a free [Hugging Face](https://huggingface.co) account and an access token (Settings → Access Tokens → create one with Write access).
2. **Create a new Space** at [huggingface.co/new-space](https://huggingface.co/new-space) — pick the **Gradio** SDK, CPU hardware (free tier).
3. Upload `webapp/app.py` and `webapp/requirements.txt` to the Space (or connect the Space to this GitHub repo).
4. In `app.py`, change `TABULAR_ADAPTER_REPO` and `IMAGE_MODEL_REPO` to the Hub repo IDs you pushed to in step 1.

**Why the numeric-measurements tab is slow:** the LLM was fine-tuned using 4-bit quantization, which requires a GPU — free Spaces only provide CPU, so that tab loads the full 3B model at full precision instead (30-60+ seconds per prediction). The image tab has no such issue since the CNN is tiny (~100K parameters) and runs instantly on CPU. If you want the LLM tab to feel fast too, that requires upgrading the Space to paid GPU hardware.
