# Evaluation Framework

Experiments for the multimodal crop health agent thesis. All scripts run from the **project root**.

---

## Directory Structure

```
evaluations/
├── exp_classification.py       # Exp 1 – CLIP vs ResNet-50 classifier benchmark
├── exp_component_analysis.py   # Exp 2 – Ablation study (RAG / CLIP / Validators)
├── exp_rag_quality.py          # Exp 3 – RAG retrieval Precision@K and MRR
├── clip_classifier.py          # CLIP classifier implementation
├── resnet_classifier.py        # ResNet-50 classifier implementation
├── eval_utils.py               # Shared utilities (Timer, EvaluationResult, metrics)
├── test_cases/
│   └── exp_test_cases.yaml     # Ground-truth test cases for component analysis
├── data/
│   ├── plantvillage/           # PlantVillage dataset (download separately, see below)
│   ├── PlantDoc-Dataset-master/# PlantDoc dataset (see DATASET_SETUP.md)
│   └── known_unknown/          # Pre-split known/unknown image sets
└── results/
    ├── classification/         # JSON outputs from exp_classification.py
    ├── component_analysis/     # JSON outputs from exp_component_analysis.py
    └── rag/                    # JSON outputs from exp_rag_quality.py
```

---

## Experiments

### Experiment 1 – Classification Benchmark (`exp_classification.py`)

Compares **CLIP** (fine-tuned, zero-shot flexible) vs. **ResNet-50** (fine-tuned, closed 38-class set) on known disease accuracy and unknown class rejection rate.

```bash
python evaluations/exp_classification.py
```

Requires PlantVillage dataset (see **Dataset Setup** below).

---

### Experiment 2 – Component Analysis (`exp_component_analysis.py`)

Ablation study evaluating how much each component contributes to output quality. Runs the agent pipeline under different configurations defined in `test_cases/exp_test_cases.yaml`.

```bash
python evaluations/exp_component_analysis.py
```

Works immediately with your existing system — no extra data needed.

**Configurations:**

| Configuration | RAG | CLIP Verify | Validators |
|---|---|---|---|
| Full System | ✅ | ✅ | ✅ |
| No RAG | ❌ | ✅ | ✅ |
| No CLIP Verify | ✅ | ❌ | ✅ |
| No Self-Correction | ✅ | ✅ | ❌ |

---

### Experiment 3 – RAG Retrieval Quality (`exp_rag_quality.py`)

Measures retrieval Precision@K and MRR with and without crop-type metadata filtering.

```bash
python evaluations/exp_rag_quality.py
```

Requires the RAG database to be populated first (`python -m app.rag.ingest --target ./data/manuals/`).

---

## Dataset Setup

### PlantVillage (required for Experiment 1)

**Option A — Kaggle CLI:**
```bash
# Set up kaggle credentials at ~/.kaggle/kaggle.json first
kaggle datasets download -d abdallahalidev/plantvillage-dataset
unzip plantvillage-dataset.zip -d evaluations/data/plantvillage
rm plantvillage-dataset.zip
```

**Option B — Manual download:**
1. Go to https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
2. Download and extract to `evaluations/data/plantvillage/`

**Expected structure:**
```
evaluations/data/plantvillage/
├── Tomato___Early_blight/     (~1000 images)
├── Tomato___Late_blight/      (~1900 images)
├── Rice___Blast/
└── ... (38 classes total, ~54k images, RGB JPG/PNG)
```

**Verify:**
```bash
ls evaluations/data/plantvillage/ | head -10
```

**Classes used by the experiment:**

| Split | Folder |
|---|---|
| Known | `Tomato___Early_blight`, `Tomato___Late_blight` |
| Unknown | `Rice___Blast` (different crop, not in CLIP training) |

For more dataset options (custom loaders, Vietnamese crop images, PlantDoc) see [`DATASET_SETUP.md`](DATASET_SETUP.md).

---

## Troubleshooting

| Error | Fix |
|---|---|
| `ImportError: No module named 'evaluations'` | Run from project root: `cd /path/to/multimodal-crop-agent` |
| `RAG engine not available` | Ingest manuals first: `python -m app.rag.ingest --target ./data/manuals/` |
| `CUDA out of memory` | `export CUDA_VISIBLE_DEVICES=""` then re-run |
| `PlantVillage directory not found` | Ensure path is `evaluations/data/plantvillage/Tomato___Early_blight/` (3 underscores) |
| `Class not found` | CLIP label `"Tomato leaf with Early blight"` maps to folder `Tomato___Early_blight` |
