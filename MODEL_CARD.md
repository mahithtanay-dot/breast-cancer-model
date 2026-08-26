# Model Card: Breast Cancer Classifier (Qwen2.5-3B + LoRA)

> **⚠️ Not for clinical use.** This model is a learning/research project. It has not been externally validated, has not undergone regulatory review, and must not be used to inform any real medical decision. See [Caveats and Recommendations](#caveats-and-recommendations).

## Model Details

- **Base model:** Qwen2.5-3B (Apache 2.0), loaded in 4-bit (QLoRA)
- **Fine-tuning method:** LoRA adapters (rank 16) on attention projections, plus a trained classification head (`AutoModelForSequenceClassification`, 2 labels)
- **Task:** Binary classification — malignant vs. benign, from numeric cytology features serialized to text
- **Developed by:** [mahithtanay-dot](https://github.com/mahithtanay-dot), personal learning project
- **Repository:** [`finetune_3B_lora.ipynb`](finetune_3B_lora.ipynb)
- **License:** inherits Qwen2.5-3B's Apache 2.0 license for the base weights; this repo's own code has no license file yet (all rights reserved by default until one is added)

## Intended Use

**In scope:** demonstrating a QLoRA fine-tuning pipeline (data prep, 4-bit quantization, PEFT, training, evaluation, inference) as an educational exercise. Useful as a reference for others learning the same pipeline, or as a baseline to compare better-suited architectures against.

**Out of scope:** any actual diagnostic, screening, or triage use involving real patients. Not a substitute for a biopsy, pathologist review, or licensed clinical judgment.

## Training Data

Two datasets are combined into one training pool, both the same benign/malignant task but with different, non-overlapping feature sets:

- [UCI Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) (`data/wdbc.data`) — 569 samples, 30 numeric features (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension — each as mean, standard error, and worst-case value) computed from digitized FNA cell nuclei images
- [UCI Wisconsin (Original)](https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original) (`data/wisconsin_original.data`) — 699 samples (683 after dropping 16 with missing values), 9 different numeric features (clump thickness, uniformity of cell size/shape, marginal adhesion, single epithelial cell size, bare nuclei, bland chromatin, normal nucleoli, mitoses)
- **Combined size:** ~1,252 samples. Rows are serialized to `key: value` text before training, which is what makes mixing two different feature schemas possible — the model reads whichever measurements are present per row rather than a fixed-width vector.
- **Labels:** malignant / benign, presumably biopsy-confirmed in both source datasets (not independently re-verified by this project)
- **Both datasets originate from the same institution and era** (University of Wisconsin Hospitals, early-to-mid 1990s) — combining them adds volume and some feature diversity, but does **not** provide independent external validation; a model that performs well here has still only ever seen data from one source population.
- **No demographic metadata** (age, ethnicity, geography) is available in either dataset, so no subgroup fairness analysis is possible with this data alone

## Evaluation

- **Split:** single random stratified split — roughly 70/10/20 train/validation/test over the ~1,252 combined rows
- **No external validation** — the model has never been tested on data from a different institution, population, or imaging setup
- **Metrics:** accuracy, F1, and AUC on the held-out test split (see the notebook's Section 8 output for current run values — not reproduced here since they vary by run and haven't been fixed to a locked evaluation protocol yet)

## Caveats and Recommendations

- **Small, single-institution dataset.** ~1,252 rows from one hospital in the 1990s (across two feature-engineering methods) is still not evidence the model generalizes to other populations, equipment, or time periods. This is the most common reason medical ML models fail when moved from a paper/notebook to the real world.
- **No external validation.** All reported metrics come from a held-out slice of the *same* dataset the model trained on, not an independent cohort.
- **Model size mismatch.** A 3-billion-parameter LLM is disproportionate to 30 numeric features and 569 rows; a properly tuned gradient-boosted tree or small MLP is the more appropriate (and more interpretable) architecture for this kind of data, and interpretability matters a lot for anything claiming clinical relevance.
- **No regulatory review.** Real diagnostic software goes through processes like FDA clearance (510(k) or De Novo, as "Software as a Medical Device") involving prospective clinical validation — none of that has happened here.
- **No uncertainty calibration.** The model's confidence scores have not been checked for calibration (i.e., a "90% confident" prediction may not actually be right 90% of the time).

This project should be read as a fine-tuning methodology exercise, not a medical tool.
