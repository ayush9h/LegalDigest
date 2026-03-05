# LegalDigest

LegalDigest for legal text understanding, built using **parameter-efficient fine-tuning (LoRA)**. The project focuses on **Indian Penal Code (IPC)** data and supports **both sequence-to-sequence and causal language models**. This repository contains a smaller-scale, reproducible version of the project.

[Finetuned Flan-t5-small HuggingFace Model Card](https://huggingface.co/ayush9h/legal-digest-flan-t5-small)

---
## Training Results

**Training Loss – Seq2Seq (FLAN-T5)**  
![Seq2Seq Loss](https://github.com/ayush9h/LegalDigest/blob/cd159e42e0f668c7e1bd39bbd177cf1971dd6a65/logs/training_loss_seq2seq.png?raw=true)

---


## Logging & Monitoring

Model logs are captured in Mlflow.

**Mlflow**
![MlFlow Dashboard](https://github.com/user-attachments/assets/a4ffb502-bb21-4d96-a46f-1e6b3a56a9e5)


Requests logs are monitored through prometheus

**Prometheus**  
![Prometheus Dashboard](https://github.com/user-attachments/assets/4adc9d1a-4439-4e7b-b042-4865921eff23)


---

## Project Structure

```
LegalDigest/
├── legal_digest/
│   ├── api/
│   │   ├── inference.py (Endpoint for interncing the finetuned model)
│   │   ├── metrics.py (Endpoint for metrics logging for Prometheus)
│   │   └── train.py (Endpoint for finetuning the model)
│   │
│   ├── config/
│   │   ├── __init__.py
│   │   └── training.yaml (Config for model finetuning)
│   │
│   ├── data/
│   │   └── constitution_qa.json
│   │
│   ├── pipelines/
│   │   ├── __init__.py
│   │   ├── inference.py (inferencing the finetuned model)
│   │   ├── metrics.py (metrics for finetuned model Rouge, f1)
│   │   ├── preprocess.py (preprocessing of the data)
│   │   └── train.py (Fine tuning process)
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── load_yaml_config.py (loads configs from training.yaml)
│   │   ├── logger.py
│   │   └── monitoring.py (monitoring uits)
│   │
│   ├── __init__.py
│   ├── app.py (streamlit demo app)
│   └── main.py (Entry point)
│
├── prometheus.yml
├── .env
├── .gitignore
├── LICENSE
├── pyproject.toml
└── README.md
```

---

## Installation and How to use

### 1. Clone the repository

```bash
git clone https://github.com/ayush9h/LegalDigest.git
cd LegalDigest
```

### 2. Create and activate virtual environment

```bash
uv venv .venv
source .venv/bin/activate        # Linux / macOS
```

```powershell
.\.venv\Scripts\Activate     # Windows
```

### 3. Install requirements

```bash
pip install -e .
```

### 3. Run fastapi and mlflow application 

In seperate terminals run the below two commands:

```bash
uvicorn main:app --reload
mlflow server
```



### 3. Open CMD and run Prometheus through Docker

Runs prometheus on localhost:9090
```bash
docker run -p 9090:9090 -v <promtheus_yml_file_path>:/etc/prometheus/prometheus.yml prom/prometheus
```

