# Model Card: Breast Cancer Classifiers

> **⚠️ Not for clinical use.** These models are a learning/research project. They have not been externally validated, have not undergone regulatory review, and must not be used to inform any real medical decision. See [Caveats and Recommendations](#caveats-and-recommendations).

This card covers two separate models — a tabular LLM classifier and an image CNN classifier. The training code and raw datasets for both live in a private companion repository (see [About the training code](README.md#about-the-training-code) in the README); this card documents what was trained, on what data, and with what known limitations, so the models can be used and reasoned about without needing access to that code.

## Model 1: Tabular Classifier (Qwen2.5-3B + LoRA)

- **Base model:** Qwen2.5-3B (Apache 2.0), loaded in 4-bit (QLoRA)
- **Fine-tuning method:** LoRA adapters (rank 16) on attention projections, plus a trained classification head (`AutoModelForSequenceClassification`, 2 labels)
- **Task:** Binary classification — malignant vs. benign, from numeric cytology features serialized to text
- **Developed by:** [mahithtanay-dot](https://github.com/mahithtanay-dot), personal learning project
- **License:** inherits Qwen2.5-3B's Apache 2.0 license for the base weights; this repo's own code has no license file yet (all rights reserved by default until one is added)

### Intended Use

**In scope:** demonstrating a QLoRA fine-tuning pipeline (data prep, 4-bit quantization, PEFT, training, evaluation, inference) as an educational exercise. Useful as a reference for others learning the same pipeline, or as a baseline to compare better-suited architectures against.

**Out of scope:** any actual diagnostic, screening, or triage use involving real patients. Not a substitute for a biopsy, pathologist review, or licensed clinical judgment.

### Training Data

Two datasets are combined into one training pool, both the same benign/malignant task but with different, non-overlapping feature sets:

- [UCI Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) — 569 samples, 30 numeric features (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension — each as mean, standard error, and worst-case value) computed from digitized FNA cell nuclei images
- [UCI Wisconsin (Original)](https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original) — 699 samples (683 after dropping 16 with missing values), 9 different numeric features (clump thickness, uniformity of cell size/shape, marginal adhesion, single epithelial cell size, bare nuclei, bland chromatin, normal nucleoli, mitoses)
- **Combined size:** ~1,252 samples. Rows are serialized to `key: value` text before training, which is what makes mixing two different feature schemas possible — the model reads whichever measurements are present per row rather than a fixed-width vector.
- **Labels:** malignant / benign, presumably biopsy-confirmed in both source datasets (not independently re-verified by this project)
- **Both datasets originate from the same institution and era** (University of Wisconsin Hospitals, early-to-mid 1990s) — combining them adds volume and some feature diversity, but does **not** provide independent external validation; a model that performs well here has still only ever seen data from one source population.
- **No demographic metadata** (age, ethnicity, geography) is available in either dataset, so no subgroup fairness analysis is possible with this data alone

### Evaluation

- **Split:** single random stratified split — roughly 70/10/20 train/validation/test over the ~1,252 combined rows
- **No external validation** — the model has never been tested on data from a different institution, population, or imaging setup
- **Metrics:** accuracy, F1, and AUC on the held-out test split (values pending a locked evaluation run — update this section once final numbers are recorded)

### Caveats

- **Small, single-institution dataset.** ~1,252 rows from one hospital in the 1990s (across two feature-engineering methods) is still not evidence the model generalizes to other populations, equipment, or time periods.
- **No external validation.** All reported metrics come from a held-out slice of the *same* dataset the model trained on, not an independent cohort.
- **Model size mismatch.** A 3-billion-parameter LLM is disproportionate to a few dozen numeric features; a properly tuned gradient-boosted tree or small MLP is the more appropriate (and more interpretable) architecture for this kind of data.
- **No uncertainty calibration.** Confidence scores have not been checked for calibration (a "90% confident" prediction may not actually be right 90% of the time).

## Model 2: Image Classifier (CNN, trained from scratch)

- **Architecture:** small custom CNN (~100K parameters) — three convolutional blocks (32/64/128 channels) with batch norm and max pooling, followed by a dropout + linear classification head
- **Task:** Binary classification — cancerous (IDC-positive) vs. non-cancerous, from a 50x50 pixel breast tissue histopathology image patch
- **Developed by:** [mahithtanay-dot](https://github.com/mahithtanay-dot), personal learning project

### Training Data

- **Source:** [Janowczyk & Madabhushi, IDC_regular_ps50_idx5](https://andrewjanowczyk.com/use-case-6-invasive-ductal-carcinoma-idc-segmentation/) — 277,524 real 50x50 pixel patches extracted from 162 patients' whole-slide breast tissue biopsy images
- **Sample used:** a balanced 25,000-patch subset (12,500 per class), sampled from the full imbalanced pool (~28% positive)
- **Labels:** cancerous (IDC-positive) / non-cancerous, as annotated in the source dataset (not independently re-verified by this project)
- **No demographic metadata** available, so no subgroup fairness analysis is possible with this data alone

### Evaluation

- **Split:** stratified 70/10/20 train/validation/test over the 25,000-patch sample
- **No external validation** — never tested on patches from a different lab, scanner, or patient population beyond the 162 patients in the source dataset
- **Metrics:** accuracy, F1, AUC, and confusion matrix on the held-out test split (values pending a locked evaluation run)

### Caveats

- **Single-source imagery.** All 162 patients came from one collection process; a model that performs well here has not been shown to generalize to different scanners, staining protocols, or patient populations.
- **Balanced sampling changes the operating point.** Real-world prevalence of IDC-positive tissue is much lower (~28% in the full dataset) than the 50/50 split used for training — a deployed version would need re-calibration or threshold adjustment for a realistic prevalence.
- **No uncertainty calibration** and **no regulatory review**, same as Model 1.

## Shared Caveats

- **No regulatory review.** Real diagnostic software goes through processes like FDA clearance (510(k) or De Novo, as "Software as a Medical Device") involving prospective clinical validation — none of that has happened for either model here.

Both models should be read as learning/methodology exercises, not medical tools.
