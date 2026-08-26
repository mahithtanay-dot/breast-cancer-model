# Model Card: Breast Cancer Classifier (Qwen2.5-3B + LoRA)

> **⚠️ Not for clinical use.** This model is a learning/research project. It has not been externally validated, has not undergone regulatory review, and must not be used to inform any real medical decision. See [Caveats and Recommendations](#caveats-and-recommendations).

## Model Details

- **Base model:** Qwen2.5-3B (Apache 2.0), loaded in 4-bit (QLoRA)
- **Fine-tuning method:** LoRA adapters (rank 16) on attention projections, plus a trained classification head (`AutoModelForSequenceClassification`, 2 labels)
- **Task:** Binary classification — malignant vs. benign, from 30 numeric cytology features serialized to text
- **Developed by:** [mahithtanay-dot](https://github.com/mahithtanay-dot), personal learning project
- **Repository:** [`finetune_3B_lora.ipynb`](finetune_3B_lora.ipynb)
- **License:** inherits Qwen2.5-3B's Apache 2.0 license for the base weights; this repo's own code has no license file yet (all rights reserved by default until one is added)

## Intended Use

**In scope:** demonstrating a QLoRA fine-tuning pipeline (data prep, 4-bit quantization, PEFT, training, evaluation, inference) as an educational exercise. Useful as a reference for others learning the same pipeline, or as a baseline to compare better-suited architectures against.

**Out of scope:** any actual diagnostic, screening, or triage use involving real patients. Not a substitute for a biopsy, pathologist review, or licensed clinical judgment.

## Training Data

- **Source:** [UCI Wisconsin Diagnostic Breast Cancer dataset](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) (`data/wdbc.data`)
- **Size:** 569 samples total, collected at a single institution, digitized from a single fine needle aspirate (FNA) imaging process, published in 1995
- **Features:** 30 numeric features (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension — each as mean, standard error, and worst-case value) computed from digitized cell nuclei images
- **Labels:** malignant (M) / benign (B), presumably biopsy-confirmed (not independently verified by this project)
- **No demographic metadata** (age, ethnicity, geography) is available in this dataset, so no subgroup fairness analysis is possible with this data alone

## Evaluation

- **Split:** single random stratified split — ~409 train / ~46 validation / ~114 test rows from the same 569-row dataset
- **No external validation** — the model has never been tested on data from a different institution, population, or imaging setup
- **Metrics:** accuracy, F1, and AUC on the held-out test split (see the notebook's Section 8 output for current run values — not reproduced here since they vary by run and haven't been fixed to a locked evaluation protocol yet)

## Caveats and Recommendations

- **Tiny, single-source dataset.** 569 rows from one hospital in the 1990s is not evidence the model generalizes to other populations, equipment, or time periods. This is the most common reason medical ML models fail when moved from a paper/notebook to the real world.
- **No external validation.** All reported metrics come from a held-out slice of the *same* dataset the model trained on, not an independent cohort.
- **Model size mismatch.** A 3-billion-parameter LLM is disproportionate to 30 numeric features and 569 rows; a properly tuned gradient-boosted tree or small MLP is the more appropriate (and more interpretable) architecture for this kind of data, and interpretability matters a lot for anything claiming clinical relevance.
- **No regulatory review.** Real diagnostic software goes through processes like FDA clearance (510(k) or De Novo, as "Software as a Medical Device") involving prospective clinical validation — none of that has happened here.
- **No uncertainty calibration.** The model's confidence scores have not been checked for calibration (i.e., a "90% confident" prediction may not actually be right 90% of the time).

This project should be read as a fine-tuning methodology exercise, not a medical tool.
