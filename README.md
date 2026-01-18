# 🎗️ Breast Cancer Detection using Artificial Neural Networks (ANN)

## 📋 Project Overview

This project implements a Deep Learning model to predict whether a breast tumor is **Malignant** (Cancerous) or **Benign** (Safe) with high accuracy (~98%).

Using the **UCI Breast Cancer Wisconsin (Diagnostic) Dataset**, the model analyzes 30 different cell features (such as radius, texture, and smoothness) to assist medical professionals in early diagnosis.

## 🚀 Key Features

* **Zero-Setup Execution:** The dataset is loaded directly from the UCI Machine Learning Repository via URL. No API keys, downloads, or local files are required.
* **Deep Learning Architecture:** Uses a multi-layer Artificial Neural Network (ANN) with Dropout regularization to prevent overfitting.
* **Data Visualization:** Includes confusion matrices and accuracy plots to evaluate performance.
* **Simulated Diagnostics:** Includes a module to simulate a "New Patient" scenario for real-time testing.

## 🧠 Model Architecture

The Neural Network is built using **TensorFlow/Keras** with the following structure:

* **Input Layer:** 30 Neurons (matches the 30 features of the dataset).
* **Hidden Layer 1:** 16 Neurons (ReLU activation) + Dropout (0.5).
* **Hidden Layer 2:** 8 Neurons (ReLU activation) + Dropout (0.5).
* **Output Layer:** 1 Neuron (Sigmoid activation) -> Returns a probability score (0-1).

## 📊 Performance

* **Test Set Accuracy:** ~97-98%
* **Loss Function:** Binary Crossentropy
* **Optimizer:** Adam

<img width="855" height="393" alt="image" src="https://github.com/user-attachments/assets/e23e917c-0eb9-47b9-8e28-f5ea41d8b160" />


## 🛠️ Tech Stack

* **Language:** Python
* **Libraries:** TensorFlow, Keras, Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn.
* **Environment:** Google Colab / Jupyter Notebook.

## 💻 How to Run

You can run this project directly in your browser without installing anything:

1. Click the **"Open in Colab"** badge at the top of this file.
2. Go to **Runtime -> Run All**.
3. Scroll to the bottom to see the **Medical Diagnostic Report** for a random patient.

---

## 📂 Code Snippet (Prediction Logic)

To ensure the model never "cheats," the diagnosis column is dropped before training. Here is how the model predicts a new patient:

```python
# Simulating a doctor's visit
patient_data = X_test[patient_id].reshape(1, -1)
prediction_prob = model.predict(patient_data)[0][0]

if prediction_prob > 0.5:
    print(f"⚠️ Diagnosis: MALIGNANT ({prediction_prob*100:.2f}% Confidence)")
else:
    print(f"✅ Diagnosis: BENIGN ({(1-prediction_prob)*100:.2f}% Confidence)")

```

---
