
# Spam Classification with Transformer: Full Pipeline

Welcome! This repository guides you through building a transformer-based language model from scratch, loading real GPT-2 weights, and fine-tuning for spam classification. The process is inspired by Sebastian Raschka's book "Build a Large Language Model From Scratch" and is designed to be beginner-friendly and practical.

---

## 🚀 What You'll Learn
- How transformers work under the hood (code from scratch)
- How to download and use official GPT-2 weights
- How to fine-tune a large language model for a custom task (spam classification)
- How to visualize and interpret training results

---

## 🛠️ Project Structure & Key Files

| File/Folder            | Purpose                                                        |
|------------------------|----------------------------------------------------------------|
| `previous_chapters.py` | Modular transformer code: dataset, model, training, generation  |
| `gpt_download.py`      | Automated GPT-2 weight download and conversion                  |
| `main.ipynb`           | End-to-end notebook: training, evaluation, visualization        |
| `accuracy-plot.pdf`    | Accuracy curve during training                                 |
| `loss-plot.pdf`        | Loss curve during training                                     |
| `requirements.txt`     | Python dependencies                                            |

---

## 🧩 Full Pipeline: Step-by-Step

### 1. Build a Transformer from Scratch
- All core components (multi-head attention, layer norm, feed-forward, etc.) are implemented in PyTorch.
- See `previous_chapters.py` for readable, well-commented code.

### 2. Download Pretrained GPT-2 Weights
- Use `gpt_download.py` to fetch official GPT-2 weights (124M, 355M, 774M, 1558M) from OpenAI or a backup mirror.
- The script checks file integrity and provides progress bars for large downloads.

**Example:**
```python
from gpt_download import download_and_load_gpt2
settings, params = download_and_load_gpt2('124M', './models')
```

### 3. Load Weights into Your Transformer
- Use the provided functions to map TensorFlow weights to your PyTorch model.
- Tokenization is handled using the GPT-2 tokenizer for compatibility.

**Example:**
```python
from previous_chapters import GPTModel, load_weights_into_gpt
model = GPTModel(settings)
load_weights_into_gpt(model, params)
```

### 4. Fine-Tune for Spam Classification
- Prepare your dataset: tokenize and batch using sliding windows.
- Train the model on labeled spam data (see `main.ipynb` for code and workflow).
- Monitor accuracy and loss using the provided plots.

---

## 🏁 Getting Started (Quickstart)

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd vasu/spam-classify
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Download GPT-2 weights:**
   ```python
   from gpt_download import download_and_load_gpt2
   settings, params = download_and_load_gpt2('124M', './models')
   ```
4. **Run the notebook:**
   Open `main.ipynb` in Jupyter or VS Code and follow the step-by-step cells.

---

## 💡 Tips for New Users
- If you are new to transformers, start by reading the comments in `previous_chapters.py`.
- The notebook (`main.ipynb`) is designed to be run top-to-bottom. Each cell builds on the previous.
- If you have trouble downloading weights, check your internet connection or use the backup mirror.
- Training can be GPU-accelerated for speed. If using CPU, expect longer runtimes for large models.
- Accuracy and loss plots (`accuracy-plot.pdf`, `loss-plot.pdf`) help you diagnose overfitting or underfitting.

---

## 📚 References & Further Reading
- [Build a Large Language Model From Scratch (Book)](https://www.manning.com/books/build-a-large-language-model-from-scratch)
- [Official Code Repository](https://github.com/rasbt/LLMs-from-scratch)

---

## ⚖️ License
Apache License 2.0 (see LICENSE.txt)

---

## 🙌 Acknowledgements
This project is based on the work of Sebastian Raschka and the open-source community. It aims to make modern NLP accessible and practical for everyone.

---

**Ready to build your own transformer and fine-tune it for spam detection? Start with the notebook and happy experimenting!**
