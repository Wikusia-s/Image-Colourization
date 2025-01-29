# Image Colorization using Autoencoder

This project implements an **image colorization** model using a **deep learning approach** with an **Autoencoder architecture**. The model takes grayscale images as input and predicts the missing color channels.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training Strategy](#training-strategy)
- [Results](#results)
- [Usage](#usage)
- [Requirements](#requirements)
- [How to Run](#how-to-run)
- [References](#references)

---

## Project Overview
The goal of this project is to train a neural network that can take grayscale images (L channel from Lab color space) and predict the missing a and b color channels. The model is trained using **Mean Squared Error (MSE)** loss or **Structural Similarity Index (SSIM)** loss.

## Dataset
The dataset consists of images converted to the **Lab color space**, where:
- **L channel** represents lightness (input to the model)
- **a, b channels** represent color information (target output)

The dataset is split into:
- **70% Training set**
- **15% Validation set**
- **15% Test set**

## Model Architecture
The model is based on an **Autoencoder** with a U-Net-like structure:
- **Encoder**: Extracts features from the grayscale input
- **Decoder**: Reconstructs the a and b channels
- **Activation Function**: `tanh` for output normalization

### Hyperparameters
- **Optimizer**: Adam
- **Loss Function**: SSIM Loss
- **Learning Rate**: 0.001
- **Epochs**: 100
- **Batch Size**: Tuned using cross-validation

## Training Strategy
- **Cross-validation** is used to find the best hyperparameters
- **Early stopping** monitors SSIM loss to prevent overfitting
- **Metrics**:
  - **Peak Signal-to-Noise Ratio (PSNR)**
  - **Structural Similarity Index (SSIM)**

### Training and Validation Metrics
After training, the following plots are generated:
1. **Loss curves** (Training & Validation)
2. **SSIM metric over epochs**
3. **PSNR metric over epochs**

## Results
The final model is evaluated by predicting color channels and reconstructing RGB images. The outputs include:
- **Original image (ground truth)**
- **Grayscale input (L channel)**
- **Predicted a and b channels**
- **Reconstructed RGB image**

## Usage
### Installation
```bash
pip install -r requirements.txt
```

## Requirements
- Python 3.x
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib

## How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/image-colorization.git
   cd image-colorization
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```


## References
- Kaggle: [Image Colorization with CNN](https://www.kaggle.com/code/basu369victor/image-colorization-basic-implementation-with-cnn)
- Research Papers on Autoencoders for Image Processing

---


