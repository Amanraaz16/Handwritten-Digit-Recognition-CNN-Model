# 🧠 Handwritten Digit Recognition using Deep Learning (ANN & CNN)

A complete **Handwritten Digit Recognition System** implemented using **Artificial Neural Networks (ANN)** and **Convolutional Neural Networks (CNN)**, trained on the **MNIST dataset**, with support for **custom handwritten image prediction** and **real-time digit drawing & recognition**.

This project was developed as part of the **Skill Enhancement Course (SEC): Introduction to Neural Networks**.

---

## 📌 Project Overview

The objective of this project is to recognize handwritten digits (0–9) using deep learning techniques.  
The system demonstrates the full pipeline of a neural network project, including:

- Training ANN and CNN models on MNIST
- Comparing ANN vs CNN performance
- Predicting digits from custom image files
- Real-time digit recognition using mouse-based drawing
- Robust preprocessing to handle real-world handwritten input

CNN is used as the **final prediction model** due to its superior accuracy in image-based tasks.

---

## 🗂️ Project Structure

```bash
Handwritten-Digit-Recognition-using-Deep-Learning
│
├── models
│   ├── mnist_ann.h5        # Trained ANN model (baseline)
│   └── mnist_cnn.h5        # Trained CNN model (main model)
│
├── dataset
│   └── (MNIST dataset loads automatically using TensorFlow)
│
├── src
│   ├── ann_model.py                # ANN training script
│   ├── cnn_model.py                # CNN training script
│   ├── predict_custom_improved.py  # Predict digits from image files
│   └── draw_predict.py             # Real-time digit drawing & prediction
│
├── digit.png
├── digit1.png
├── digit2.png
├── digit3.png
├── digit4.png
├── digit5.png
├── digit6.png
├── digit7.png
├── digit8.png
├── digit9.png
│
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation

```


---

## 🧪 Models Implemented

### 🔹 Artificial Neural Network (ANN)
- Input Layer: 784 neurons (28×28 flattened image)
- Hidden Layers: 128, 64 neurons (ReLU activation)
- Output Layer: 10 neurons (Softmax)
- Accuracy: ~97.7%
- Purpose: Baseline model for understanding neural networks

### 🔹 Convolutional Neural Network (CNN)
- Convolutional layers with ReLU activation
- MaxPooling layers for dimensionality reduction
- Dense layers for classification
- Output Layer: 10 neurons (Softmax)
- Accuracy: ~98.9%
- Used for all final predictions and real-time recognition

---

## 🛠️ Image Preprocessing Pipeline

Real-world handwritten digits do not directly match MNIST format.  
To handle this, a preprocessing pipeline is applied before prediction:

1. Convert image to grayscale
2. Noise reduction using Gaussian Blur
3. Adaptive thresholding (binary inversion)
4. Contour detection to isolate the digit
5. Resize digit to 20×20 while maintaining aspect ratio
6. Pad image to 28×28 pixels
7. Center digit using center-of-mass alignment
8. Normalize pixel values to range [0, 1]

This ensures compatibility with the MNIST-trained CNN model.

---

## 🎨 Real-Time Digit Drawing & Prediction

The `draw_predict.py` script enables real-time handwritten digit recognition.

### Features:
- Draw digits using mouse
- Predict digit using trained CNN
- Display prediction confidence
- Save processed 28×28 images for debugging

### Keyboard Controls:
- **P** → Predict digit
- **C** → Clear drawing canvas
- **S** → Save processed digit image
- **Q** → Quit application

---

## 🚀 How to Run the Project

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 2️⃣ Train the Models
```bash
python src/ann_model.py
python src/cnn_model.py
```

### 3️⃣ Predict Digit from an Image
```bash
python src/predict_custom_improved.py digit.png
```

### 4️⃣ Run Real-Time Drawing Application
```bash
python src/draw_predict.py
```

## 📊 Evaluation Metrics

Accuracy – Percentage of correctly predicted digits

Softmax Probabilities – Confidence score for each digit class

CNN outperforms ANN due to its ability to learn spatial features such as edges, curves, and shapes

## 📚 Technologies Used

- Python

- TensorFlow / Keras

- OpenCV

- NumPy

- MNIST Dataset

- Git & GitHub

## 🎨 Real-Time Digit Drawing & Prediction

The application allows users to draw digits using the mouse and predict them in real time using a trained CNN model.

## 🎨 Real-Time Digit Drawing & Prediction

<img width="200" height= "200" alt="Real-Time Drawing Interface" src="https://github.com/user-attachments/assets/f6ecb700-7761-44c7-8646-be6acde276f1" />


## 🎯 The predicted digit and confidence score are displayed instantly.

<img alt="Real-Time Drawing Interface" src="https://github.com/user-attachments/assets/9fd35716-c2a4-4792-b8a4-f7f690d17bc7" />


## 🎓 Academic Relevance

This project covers major concepts from the Introduction to Neural Networks syllabus, including:

Artificial Neural Networks (ANN)

Convolutional Neural Networks (CNN)

Activation Functions (ReLU, Softmax)

Loss Functions and Optimizers

Backpropagation

Image Preprocessing

Model Training, Evaluation, and Deployment

## 🏁 Conclusion

This project demonstrates a complete deep learning pipeline for handwritten digit recognition, from model training to real-time deployment. By combining ANN and CNN models with a robust preprocessing pipeline, the system accurately predicts both standard MNIST digits and real-world handwritten inputs.

## 👨‍💻 Author

Aman Raj

B.Tech (CSE)

Skill Enhancement Course – Introduction to Neural Networks
