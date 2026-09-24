# FCIH: Fine-Grained Cross-View Interaction for Heterogeneous Graph-Based Learning Resource Recommendation

This repository provides the official implementation of FCIH for heterogeneous graph-based learning resource recommendation. FCIH combines adaptive multi-hop information fusion with fine-grained cross-view interaction to learn informative node representations from heterogeneous relational data.

## 📌 Overview

**FCIH** constructs a multi-relational heterogeneous view and a target-relation view. Within each view, the Adaptive Multi-hop Information Fusion Module (AMIFM) combines edge-type-guided node-level attention, global edge-type weighting, and adaptive multi-hop fusion. The resulting view-specific representations are then integrated through node-wise adaptive residual fusion.

The experiments are conducted on Semantic Scholar, MOOCCube, and AAN using nDCG@10, nDCG@20, and MAP as the main evaluation metrics.

## ⚙️ Requirements

Ensure the following packages are installed:

```bash
Python >= 3.10
PyTorch
transformers
torch-scatter
networkx
numpy
scikit-learn
tqdm
pandas
matplotlib
```

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

## 📁 Project Structure

```text
.
├── AAN/
│   ├── train_fcih.py
│   └── run_multiseed.sh
│
├── SemanticScholar/
│   ├── train_fcih.py
│   └── run_multiseed.sh
│
├── MOOCCube/
│   ├── train_fcih.py
│   └── run_multiseed.sh
│
├── requirements.txt
└── README.md
```

## 🔤 Node Features

Node features are initialized using 768-dimensional `[CLS]` representations from the final hidden layer of pretrained BERT models. The feature generation code supports pretrained models such as `bert-base-uncased` and BERT-base Chinese, with a maximum input length of 64.

## 🚀 Running the Model

1. Place the dataset file in the corresponding `data/` directory, or provide its location with `--data_path`.



2. Run training and evaluation. For example:

```bash
cd AAN
python train_fcih.py 
```
3. To run all ten seeds used in the experiments:

```bash
bash run_multiseed.sh 
```

## 📊 Evaluation Metrics

The main reported metrics are:

- nDCG@10
- nDCG@20
- MAP
