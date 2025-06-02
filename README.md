# Multi-Modal Visual Question Answering (VQA)

A deep learning-based solution for Visual Question Answering (VQA) that combines image and textual data to determine the correct answer from multiple candidates. This project explores various vision-text matching models and provides a modular framework for training and evaluation.

## Overview

Visual Question Answering is a multi-modal machine learning task that combines computer vision and natural language processing. This project formulates the problem as a classification task and evaluates multiple models that predict whether a given image-question-answer triplet is a valid match.

Supported model types include:

- Convolutional Neural Networks (CNNs)
- Vision Transformers (ViTs)
- Custom CNN Architectures

## Features

- Modular and extensible PyTorch-based training pipeline
- Support for both CNN and Transformer-based vision models
- Text encoding using precomputed embeddings
- Custom dataset loader for image-question-answer triplets
- Evaluation with:
  - Accuracy
  - Balanced Accuracy
  - Confusion Matrix
  - Mean Reciprocal Rank (MRR)
- Optional visualization tools for loss curves and confusion matrices

## Directory Structure

```
.
├── model_for_camparison.py     # Train and compare multiple models
├── multi_choice_vqa.py         # Focused script for CNN baseline
├── testing.py                  # Load and evaluate trained model
├── utils/                      # (optional) utilities for preprocessing
├── data/                       # Place your dataset and embeddings here
└── README.md
```

## Installation

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Dataset Format

This project expects a dataset structured as follows:

- A folder of images (e.g., `.jpg`, `.png`)
- A file containing image-question-answer-label triplets (e.g., `.txt`, `.csv`)
- A pickle file with precomputed text embeddings for questions and answers

Example input line:
```
image1.jpg    What is the object?    A chair    match
```

## Usage

### Train and evaluate all models:
```bash
python model_for_camparison.py
```

### Train only CNN baseline:
```bash
python multi_choice_vqa.py
```

### Evaluate a trained model:
```bash
python testing.py
```

## Outputs

- Training loss plot
- Confusion matrix heatmap
- Accuracy and balanced accuracy metrics

## Customization

- Swap out text embeddings with your preferred models (e.g., BERT, T5)
- Add new model types by extending the `ITM_Model` class
- Adjust hyperparameters such as learning rate, epochs, batch size

## License

This project is open-source under the MIT License.

## Author

**Faisal Hussain**  
GitHub: [@raifaisalhussain](https://github.com/raifaisalhussain)  
LinkedIn: [linkedin.com/in/raifaisalhussain](https://www.linkedin.com/in/raifaisalhussain)
