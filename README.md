# Bangla Sentiment Analysis using LLaMA 3.1

This project implements sentiment analysis for Bangla text using LLaMA 3.1, addressing the challenges of a small, imbalanced dataset containing mixed Bangla and Banglish text.

## Project Overview

The project analyzes sentiment in Bangla text, classifying it into three categories: positive, neutral, and negative. It demonstrates how to handle mixed-language text processing, data augmentation, and fine-tuning of large language models for low-resource languages.

## Dataset

- Initial dataset: 99 labeled samples
  - 45 neutral
  - 42 negative
  - 12 positive
- Average word count: ~43.44 words per sample
- Language composition: 3,806 Bangla words and 434 Banglish words (8.77:1 ratio)

## Repository Structure
├── notebooks/
│ ├── Task_Assignments_ML_Engineer_Role_Part1.ipynb # Dataset overview and basic training
│ ├── Task_Assignments_ML_Engineer_Role_Part2.ipynb # Preprocessing and FastText
│ ├── Task_Assignments_ML_Engineer_Role_Part3.ipynb # Augmentation with backtranslation
│ ├── LLama3.1_Sentiment_after_preprocessing.ipynb # LLaMA analysis on preprocessed data
│ └── LLama3.1_Sentiment_before_preprocessing.ipynb # LLaMA analysis on raw data
├── datasets/
│ ├── Dataset-Sentiment_Analysis.xlsx # Original dataset
│ ├── Preprocess_dataset.csv # Cleaned dataset
│ └── balanced_dataset_v1.csv # Augmented dataset
└── Report.txt # Detailed analysis report

## Methodology

1. **Preprocessing**
   - Retained Banglish words due to small dataset size
   - Mapped emojis to Bangla equivalents
   - Removed unnecessary elements (whitespace, special characters, numbers)
   - Applied Bangla stemming
   - Removed noise like "See translation" artifacts

2. **Data Augmentation**
   - Implemented SMOTE for initial balancing
   - Applied back-translation using googletrans
   - Achieved balanced distribution (45 samples per class)

3. **Model Architecture**
   - Utilized LLaMA 3.1 (8B parameters)
   - Applied 4-bit quantization
   - Implemented PEFT with LoRA
   - Used custom prompts for training

## Results

The final model achieved:
- Overall accuracy: 93%
- Perfect classification (100%) for negative and neutral samples
- 85.7% accuracy for positive samples
- High precision and recall across all categories

## Challenges Addressed

- Small dataset size and class imbalance
- Mixed-language text processing
- Model selection for limited data
- VRAM optimization through quantization

## Requirements

- Python 3.x
- PyTorch
- transformers
- PEFT
- BitsAndBytes
- googletrans
- pandas
- numpy
- scikit-learn

## Installation

```bash
git clone https://github.com/yourusername/bangla-sentiment-analysis.git
cd bangla-sentiment-analysis
pip install -r requirements.txt
