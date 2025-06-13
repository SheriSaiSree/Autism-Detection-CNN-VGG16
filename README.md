# Autism-Detection-CNN-VGG16
A deep learning model using VGG16 to classify facial images as Autistic or Non-Autistic.

Autism Detection Using CNN (VGG16)
This project implements a convolutional neural network model based on VGG16 architecture to classify facial images into two categories: Autistic and Non-Autistic. The model leverages transfer learning using pre-trained ImageNet weights and data augmentation to improve generalization.

Directory Structure
Autism-Detection-CNN-VGG16/
│
├── train/                  # Training images
│   ├── Autistic/
│   └── Non_Autistic/
│
├── valid/                  # Validation images
│   ├── Autistic/
│   └── Non_Autistic/
│
├── test/                   # Testing images
│
├── autism_detection.ipynb  # Jupyter Notebook with full model pipeline
├── README.md               # Project documentation (this file)

Dataset Description
The dataset is organized into three folders:

train/ – for training the model

valid/ – for validating the model performance during training

test/ – for final evaluation of the model

Each folder contains two subdirectories:

Autistic/

Non_Autistic/

Technologies Used
Python

TensorFlow / Keras

OpenCV

Seaborn & Matplotlib (visualizations)

Pre-trained VGG16 (Transfer Learning)

Scikit-learn (Metrics and Evaluation)

 Model Pipeline
Image Preprocessing
Resize to 150x150x3
Label encoding: 0 for Non-Autistic, 1 for Autistic
Normalize using ImageDataGenerator
Model Architecture
Base model: VGG16 (frozen layers)

Custom classifier head:
Flatten
Dense (512, ReLU)
Dropout (0.5)
Dense (1, Sigmoid)

Training Configuration
Loss: binary_crossentropy
Optimizer: Adam
Metrics: accuracy
Epochs: 9
Batch size: 32

Evaluation
Accuracy: 73.67%
F1-Score: 0.74

Visualized:
Accuracy/Loss curves
Confusion Matrix
ROC Curve

Results
Metric	Score
Accuracy	73.67%
Precision	0.74
Recall	0.73
F1-Score	0.74
ROC-AUC	0.74

Highlights
Achieved strong results using transfer learning.
Balanced classification performance between classes.
Visualization of predictions on test samples.
Reproducible pipeline from preprocessing to evaluation.

Future Improvements
Fine-tune deeper VGG16 layers.
Try newer models (e.g., ResNet50, EfficientNet).
Experiment with more epochs and LR schedules.


Author
Saisree Sheri
