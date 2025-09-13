# Deep Learning Models on MNIST & IMDB Datasets

This repository contains implementations of three deep learning models using TensorFlow/Keras:

1. **Feed-Forward Neural Network (FNN)** for handwritten digit classification (MNIST).
2. 
3. **Convolutional Neural Network (CNN)** for image classification (MNIST).
4. 
5. **Recurrent Neural Network (RNN/LSTM)** for sentiment analysis (IMDB Movie Reviews).  

---

## 📌 Requirements

Make sure you have the following installed:

- Python 3.8+
- TensorFlow 2.x
- NumPy
- Matplotlib

## 📂 Datasets Used

MNIST: Handwritten digits dataset (28x28 grayscale images, 10 classes).

IMDB: Movie review dataset for sentiment analysis (positive/negative).

Both datasets are included in tensorflow.keras.datasets and will be downloaded automatically when you run the code.

## 🚀 Implementations

1. Feed-Forward Neural Network (FNN)

Input: Flattened MNIST images (28×28).

Two hidden layers with ReLU activation.

Output: 10 neurons with Softmax activation for digit classification.

Trained for 3 epochs.

👉 Achieves ~97% test accuracy.

2. Convolutional Neural Network (CNN)

Input: MNIST images reshaped to (28, 28, 1).

Convolutional + MaxPooling layers for feature extraction.

Dropout for regularization.

Dense layers with ReLU and final Softmax for classification.

Trained for 3 epochs.

👉 Achieves ~99% test accuracy.

Also includes feature map visualization of convolutional layers.

3. Recurrent Neural Network (RNN/LSTM)

Input: IMDB movie reviews, tokenized and padded to length 200.

Embedding layer (128 dimensions).

LSTM layer with 128 units.

Dense output with Sigmoid for binary classification.

Trained for 3 epochs.

👉 Achieves ~85–88% test accuracy.

## 📊 Training & Evaluation

For each model:

Training & validation accuracy/loss plots are generated using Matplotlib.

Models are evaluated on test data.

Predictions and sample outputs are displayed.

## 🖼️ Sample Outputs
Accuracy & Loss Plots

FNN & CNN: Training vs Validation Accuracy/Loss on MNIST.

RNN (LSTM): Training vs Validation Accuracy/Loss on IMDB.

CNN Feature Maps

Visualization of intermediate feature maps from the first convolutional layer.

## ▶️ Usage

Code: code.py
