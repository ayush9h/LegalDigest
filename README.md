# LegalDigest

LegalDigest for legal text understanding, built using **parameter-efficient fine-tuning (LoRA)**. The project focuses on **Indian Penal Code (IPC)** data and supports **both sequence-to-sequence and causal language models**. This repository contains a smaller-scale, reproducible version of the project.

---
## Training Results

**Training Loss – Seq2Seq (FLAN-T5)**  
![Seq2Seq Loss](https://github.com/ayush9h/LegalDigest/blob/cd159e42e0f668c7e1bd39bbd177cf1971dd6a65/logs/training_loss_seq2seq.png?raw=true)

**Training Loss – Causal LM (LLaMA)**  
![Causal Loss](https://github.com/ayush9h/LegalDigest/blob/cd159e42e0f668c7e1bd39bbd177cf1971dd6a65/logs/training_loss_causal.png?raw=true)

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/ayush9h/LegalDigest.git
cd LegalDigest
```

### 2. Create and activate virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
```

```powershell
.\.venv\Scripts\Activate.ps1     # Windows
```

### 3. Install project (editable mode)

```bash
pip install -e .
```

---

## Project Structure

```
LegalDigest/
├── legal_digest/
│   ├── config/
│   ├── core/
│   ├── pipelines/
│   └── utils/
├── data/
│   ├── raw/
│   └── processed/
├── checkpoints/
├── logs/
├── pyproject.toml
└── README.md
```

---

## Data Preparation


Run preprocessing:

```bash
python -m legal_digest.pipelines.preprocess
```

Output:

```
data/processed/ipc_train.jsonl
```

---

## Training

### Train FLAN-T5

```bash
legal-digest-train flan_t5
```

### Train LLaMA

```bash
legal-digest-train llama
```

---

## Logging & Monitoring

Logs are written to console and rotating files:

```
logs/
├── preprocess.log
├── train_flan_t5.log
├── train_llama.log
```
---
## Configuration

All behavior is controlled via YAML files:

- model.yaml
- training.yaml
- data.yaml
