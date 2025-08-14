# End-to-End Transformer for Text Classification

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?style=for-the-badge&logo=pytorch)
![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models%20%7C%20Spaces-yellow?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)

An end-to-end project building a Transformer from scratch and fine-tuning DistilBERT for text classification, achieving **94.8% test accuracy** and deployed as a live, interactive demo on Hugging Face Spaces.

---

### ► Try the Live Demo!

This project culminates in a deployed Gradio application. You can test the fine-tuned model's performance in real-time.

**[https://huggingface.co/spaces/nabeelshan/distilbert-agnews-classifier](https://huggingface.co/spaces/nabeelshan/distilbert-agnews-classifier)**

![Gradio Demo GIF](https://huggingface.co/spaces/nabeelshan/distilbert-agnews-classifier/resolve/main/demo.gif)
*(A live demo of the final classifier, built with Gradio and deployed on Hugging Face Spaces)*

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Key Features](#2-key-features)
3. [Final Results & Analysis](#3-final-results--analysis)
4. [Tech Stack](#4-tech-stack)
5. [Repository Structure](#5-repository-structure)
6. [Setup & Usage](#6-setup--usage)

---

## 1. Project Overview

This repository documents a comprehensive investigation into Transformer-based models for text classification on the AG News dataset. The project was executed in two main phases:

1.  **Phase 1: Build from Scratch:** A custom Transformer model was implemented from the ground up in PyTorch to gain a fundamental understanding of the architecture, including custom positional encodings and multi-head self-attention layers. This model was trained over **160+ epochs** using a **multi-stage optimization** strategy.

2.  **Phase 2: Fine-Tune a SOTA Model:** A pre-trained DistilBERT model was fine-tuned to leverage the power of transfer learning. This phase focused on the practical application of state-of-the-art tools from the Hugging Face ecosystem to achieve maximum performance.

The project covers the full MLOps lifecycle: data preprocessing, model architecture design, multi-phase training, iterative optimization, final evaluation, and deployment as an interactive web application.

---

## 2. Key Features

- **Custom Transformer Implementation:** A complete, **from-scratch implementation** of a Transformer Encoder in PyTorch.
- **Advanced Training Techniques:** Multi-stage training pipeline with checkpointing, resumption, and experimentation with various optimizers **(SGD, Adam)** and learning rate schedulers (`StepLR`, `CosineAnnealingLR`).
- **State-of-the-Art Fine-Tuning:** Efficiently fine-tuned DistilBERT using the Hugging Face `transformers` library, achieving a 94.8% test accuracy.
- **Comprehensive Evaluation:** Rigorous, multi-phase evaluation with detailed performance logging and visualization.
- **Live Interactive Demo:** The final model is deployed as a user-friendly Gradio application on Hugging Face Spaces, making the model's capabilities tangible and accessible.

---

## 3. Final Results & Analysis

The two primary models were evaluated on the same held-out test set. The fine-tuned DistilBERT model demonstrated a significant performance improvement over the custom model trained from scratch, highlighting the immense power of transfer learning.

| Model Architecture | Best Validation Accuracy | Final Test Accuracy |
| :--- | :---: | :---: |
| Custom Transformer (From Scratch) | 93.09% | 90.32% |
| **Fine-Tuned DistilBERT** | **94.59%** | **94.79%** |

### Performance Visualization

<img src="outputs/final_model_comparison_detailed.png" alt="Final Model Performance Comparison" height=500 width="1000"/>

The final comparison shows a **+4.47% absolute improvement in test accuracy** from using a fine-tuned model. This is because the pre-trained model has already learned a deep, nuanced understanding of the English language from a massive corpus, which it can then adapt to our specific task with minimal training.

<img src="outputs/comparative_performance.png" alt="Comparative Performance Bar Chart" width="1000"/>
<img src="outputs/learning_curves.png" alt="Full Training Journey Learning Curves" height=500 width="1000"/>

The learning curves for the custom model show a clear story of iterative improvement: an initial learning phase, followed by aggressive fine-tuning with Adam, and finally a "polishing" phase with `CosineAnnealingLR` to achieve the best possible performance.

---

## 4. Tech Stack

- **Frameworks & Libraries:** PyTorch, Hugging Face (Transformers, Tokenizers, Hub), Gradio, Scikit-learn
- **Data Science & MLOps:** Pandas, NumPy, Matplotlib, Seaborn
- **Tools:** Git & GitHub, Jupyter Notebooks

---

## 5. Repository Structure

The repository is organized to professional standards for clarity, reproducibility, and scalability, clearly separating source code, experimental notebooks, data, and outputs.

```
.
├── 📁 data/                  # Raw AG News dataset files
│   └── 📁 ag_news/
│       ├── test.parquet
│       └── train.parquet
│
├── 📁 notebooks/              # Jupyter notebooks for experimentation and analysis
│   ├── 01_Custom_Transformer_Initial_Training.ipynb
│   ├── 02_Custom_Transformer_Phase_A.ipynb
│   ├── ... (Sequential training phases)
│   ├── 08_Fine_Tuning_DistilBERT.ipynb
│   └── 09_Results_Analysis_and_Visualization.ipynb
│
├── 📁 outputs/                # All generated artifacts from the notebooks
│   ├── final_evaluation_summary.csv
│   ├── learning_curves.png
│   └── final_model_comparison_detailed.png
│
├── 📁 src/                    # Core, reusable Python source code
│   ├── dataloader.py         # Data loading and preprocessing pipeline
│   └── model.py              # Custom Transformer architecture and training functions
│
├── 📄 README.md               # You are here!
└── 📄 requirements.txt        # Python dependencies for setting up the environment

---

## 6. Setup & Usage

To replicate this project, follow these steps:

### Prerequisites
- Python 3.10+
- PyTorch 2.0+
- Access to a GPU is recommended for training.

### 1. Clone the Repository
```bash
git clone https://github.com/nabeelshan78/Transformer-AGNews-Classifier.git
cd Transformer-AGNews-Classifier
```

### 2. Set Up the Environment
Create a virtual environment and install the required packages.
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
pip install -r requirements.txt

```

### 3. Download the Dataset
The AG News dataset can be downloaded and placed in the `data/ag_news` directory.

### 4. Run the Notebooks
The notebooks are located in the `/notebooks` directory and are numbered to show the complete, iterative development process.

- **`01_...`**: The initial training and development of the custom Transformer model from scratch (Epochs 1-50).
- **`02_` to `06_...`**: A series of notebooks representing the sequential fine-tuning phases of the custom Transformer (Epochs 51-160).
- **`07_...`**: A notebook for running sample predictions with the final custom model.
- **`08_...`**: The complete fine-tuning pipeline for the state-of-the-art DistilBERT model.
- **`09_...`**: The final script that loads all experimental results and generates the comparison plots.
---
