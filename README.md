# End-to-End NLP: From-Scratch Transformer vs. Deployed DistilBERT

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?style=for-the-badge&logo=pytorch)
![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models%20%7C%20Spaces-yellow?style=for-the-badge)
![Gradio](https://img.shields.io/badge/Gradio-Deployed%20Demo-FF7600?style=for-the-badge&logo=gradio)

</div>

To demonstrate mastery over the full NLP development lifecycle, I undertook a comprehensive project on the AG News dataset. I began by **engineering a Transformer classifier from first principles** in PyTorch to build a deep, architectural understanding. I then **fine-tuned a state-of-the-art DistilBERT model**, leveraging transfer learning to achieve **94.8% test accuracy**.

The project culminates in a deployed, interactive web application on **Hugging Face Spaces**, showcasing my ability to handle the entire MLOps workflow from initial research and implementation to final production deployment.

---

## 🚀 Live Demo & Model

Interact with the final deployed model and see the code in action.

| Resource | Link |
| :--- | :--- |
| **Interactive Live Demo** | **[🚀 Hugging Face Spaces](https://huggingface.co/spaces/nabeelshan/distilbert-agnews-classifier)** |
| **Fine-Tuned Model Card** | **[📦 Hugging Face Hub](https://huggingface.co/nabeelshan/distilbert-finetuned-agnews)** |

---

## 🏆 My Key Achievements

* **Engineered a Transformer Classifier from Scratch:** I implemented a complete `TransformerEncoder`-based classifier in PyTorch, including a custom `PositionalEncoding` layer, to solidify my understanding of the architecture's core mechanics.
* **Executed a Multi-Stage, 160-Epoch Training Strategy:** For the custom model, I designed and executed a systematic training regimen, intelligently switching optimizers (SGD to Adam) and learning rate schedulers (`StepLR` to `CosineAnnealingLR`) to maximize performance, achieving **90.3%** test accuracy.
* **Fine-Tuned and Deployed a SOTA Model (DistilBERT):** I leveraged the Hugging Face ecosystem to fine-tune DistilBERT, achieving a superior **94.8% test accuracy in just 3 epochs**. I then successfully containerized and deployed this model as a live Gradio application.
* **Designed an Optimized Data Processing Pipeline:** I built an efficient data loader that sorts input sequences by length before batching. This technique **dramatically minimizes padding**, increasing GPU utilization and accelerating training throughput.
* **Conducted In-Depth Comparative Analysis:** I performed a head-to-head evaluation of the two models, creating detailed visualizations to analyze the trade-offs between training from scratch and leveraging transfer learning.

---

## 📊 Results & Comparative Analysis

My analysis revealed that the fine-tuned DistilBERT model significantly outperformed the custom model, providing a **+4.47% absolute improvement in test accuracy** with over **50x less training time**. This powerfully demonstrates the efficiency and effectiveness of transfer learning in modern NLP.

| Model Architecture | Best Validation Accuracy | Final Test Accuracy | Training Time |
| :--- | :---: | :---: | :---: |
| Custom Transformer (From Scratch) | 93.09% | 90.32% | ~160 Epochs |
| **Fine-Tuned DistilBERT** | **94.59%** | **94.79%** | **3 Epochs** |

<p align="center">
<img src="outputs/final_model_comparison_detailed.png" alt="Final Model Performance Comparison" width="800"/>
<em>Figure 1: Detailed comparison of accuracy and loss metrics for both models on the test set.</em>
</p>

<p align="center">
<img src="outputs/learning_curves.png" alt="Full Training Journey Learning Curves for Custom Model" width="800"/>
<em>Figure 2: The 160-epoch training journey of the custom model, showing the impact of switching optimizers and learning rate schedulers.</em>
</p>

---

## 🛠️ Tech Stack

* **Frameworks & Libraries:** PyTorch, Hugging Face (Transformers, Tokenizers, Hub), Gradio, Scikit-learn
* **Data Science & MLOps:** Pandas, NumPy, Matplotlib, Seaborn
* **Tools:** Git & GitHub, Jupyter Notebooks, VS Code

---

## ⚙️ Architectural Highlights (From-Scratch Model)

I designed the custom model with a clean, modular structure. Below are key components I implemented.

<details>
<summary><strong>Click to see the Custom Positional Encoding Implementation</strong></summary>

```python
# src/model.py: PositionalEncoding
# I implemented the classic sine/cosine positional encoding to inject sequence order
# information into the token embeddings.
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, vocab_size=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(vocab_size, d_model)
        position = torch.arange(0, vocab_size, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
````
</details>

<details>
<summary><strong>Click to see the Optimized Data Collation Logic</strong></summary>

```python
# src/dataloader.py: collate_batch function
# By sorting data by length beforehand, this function pads sequences only to the
# max length within a batch, not the entire dataset, significantly speeding up training.
def collate_batch(batch):
    label_list, text_list, len_list = [], [], []
    for (_label, _text) in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)

    labels = torch.tensor(label_list, dtype=torch.int64)
    texts = pad_sequence(text_list, batch_first=True) # Key optimization
    return labels.to(DEVICE), texts.to(DEVICE)
```
</details>

---

## 🚀 Getting Started

To replicate this project and my results, follow these steps.

**Prerequisites:**
* Python 3.10+
* PyTorch 2.0+
* A CUDA-enabled GPU is highly recommended for reasonable training times.

### 1. Clone the Repository
```bash
git clone https://github.com/nabeelshan78/NLP-From-Scratch-to-Deployment.git
cd NLP-From-Scratch-to-Deployment
```

### 2. Set Up a Virtual Environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

### 3. Run the Experiments
The notebooks in the `/notebooks` directory document my entire workflow.

* **Custom Transformer**: Run notebooks `01` through `06` sequentially to replicate the multi-stage training.
* **DistilBERT Fine-Tuning**: Run notebook `08` for the complete fine-tuning and evaluation pipeline.

---

## 📂 Repository Structure
```
NLP-From-Scratch-to-Deployment/
├── 📁 data/                  # Raw AG News dataset files
├── 📁 notebooks/             # Jupyter notebooks for each experimental phase
├── 📁 outputs/               # Generated charts, logs, and model artifacts
├── 📁 src/                   # Core Python source code
│   ├── dataloader.py        # Optimized data processing pipeline
│   └── model.py             # Custom Transformer architecture
├── 📄 .gitignore
├── 📄 README.md
└── 📄 requirements.txt
```


