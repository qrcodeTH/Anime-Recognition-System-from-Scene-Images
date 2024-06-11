# Anime Recognition System from Scene Images (AniRec 🔎)
This repository contains the code for training a classification/retrieval model to identify anime titles from scene images. 

# Overview
The goal of this project is to address the issue of identifying anime titles from images posted on the internet. This system enables users to upload an image of an anime scene and receive the corresponding anime title.

# Features
• Classification Model: Classifies anime based on scene images.

• Retrieval Model: Retrieves the most similar anime titles for given scene images.

# Dataset
Dataset drive : https://drive.google.com/file/d/1adYN9lmDqZ0vXpLLiJJ1kvilKKUazvdw/view?usp=sharing

# Models
This repository includes implementations of several popular neural network architectures for efficient image classification and retrieval:

• EfficientNet-B0

• EfficientNet-B7

• ResNeXt-101

• SEResNeXt-101

• MobileNetV2

# Loss Functions
A custom label smoothing loss function is implemented to improve model generalization:

• Label Smoothing Loss

# Training and Validation
The main.py script includes the training and validation pipeline, utilizing data loaders, augmentation techniques, and model checkpoints to facilitate efficient training.

# Feature Extraction
The feature_extraction.py script is designed to extract deep features from the trained models, which can then be used for image retrieval tasks.

# Image Retrieval
The retrieve.py script demonstrates how to perform image retrieval using extracted features and a query image. It displays the most similar images from the dataset.

# Evaluation Metrics
The classification_evaluation.py and retrieval_evaluation.py scripts provide a comprehensive evaluation of the models using the following metrics:

• Accuracy

• Top-5 Accuracy

• Mean Reciprocal Rank (MRR)

• Mean Average Precision (MAP)

# Getting Started
Requirements

• Python 3.x

• PyTorch

• torchvision

• numpy

• scikit-learn

• PIL

• tqdm

# Installation
1. Clone the repository:
```
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
```

2. install the required dependencies:
```
pip install -r requirements.txt
```
# Usage
## Training
To start training a model, run:
```
python main.py --train
```
## Feature Extraction
To extract features from a trained model, run:
```
python feature_extraction.py
```
## Image Retrieval
To perform image retrieval, run:
```
python retrieve.py --query <path_to_query_image>
```
## Evaluation
To evaluate the model performance on classification tasks, run:
```
python classification_evaluation.py
```
To evaluate the model performance on retrieval tasks, run:
```
python retrieval_evaluation.py
```
# Blog
For detailed explanations and insights about this project, visit my medium blog(Thai).

# Deployment
Website : https://anime-recognition-system-from-scene-images-zvyniwrjmg8tpz5e8vh.streamlit.app/
