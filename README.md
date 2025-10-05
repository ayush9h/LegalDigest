# LegalDigest

LegalDigest project utilizes a fine-tuned sequence-to-sequence model based on Google's FLAN-T5-Small architecture, specifically trained on Indian Penal Code (IPC) sections dataset. This repository contains a smaller-scale, reproducible version of the project.

## Features

- **Seq2Seq Architecture**: Leverages transformer-based sequence-to-sequence modeling for high-quality text generation
- **IPC-Specialized**: Fine-tuned specifically on Indian Penal Code sections for domain-specific accuracy
- **Efficient Model**: Built on FLAN-T5-Small for optimal performance with reasonable computational requirements


# Screenshots
![Loss](https://github.com/ayush9h/LegalDigest/blob/main/logs/training_loss_seq2seq.png)
![Loss](https://github.com/ayush9h/LegalDigest/blob/main/logs/training_loss_causal.png)


## Installation

1. Clone the repository:
```bash
git clone https://github.com/ayush9h/LegalDigest.git
cd LegalDigest
```

2. Install dependencies:
```bash
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

3. Download or prepare your legal dataset (IPC sections format recommended)


## Project Structure

```
LegalDigest/
├── data/
│   ├── preprocessing.py    # Transforming the training data
│   ├── ipc.jsonl           # Training data(.jsonl)
│   └── IPC.csv             # Training data(.csv)
├── src/
│   ├── config.py           # Model config script
│   ├── train.py            # Training script
│   ├── model.py            # Huggingface model loading script
│   └── preprocess.py       # Data tokenization script
├── requirements.txt         # Project dependencies
├── README.md               # Project documentation
└── main.py                 # Entry point
```

## Future Work
- Real-time web interface
- Improvement in summary generation


