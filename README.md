
# 🧠 Brain Tumor Detection Web App

A deep learning-based web application for classifying brain MRI images as **Tumor** or **Normal** using a Convolutional Neural Network (CNN). The project features a user-friendly interface built with Flask, allowing users to upload an image and receive an instant diagnosis.

---

## 🚀 Features

- ✅ Classify brain MRI images as **Tumor** or **Normal**
- 🧠 Uses a trained **CNN** model with Keras and TensorFlow
- 🌐 Web interface built using **Flask**
- 🖼️ Upload your own MRI scans for prediction
- 📦 Easy-to-run Python project

---


## 🧪 How to Run the App

1. **Clone the Repository**

2. **Install Dependencies**

Make sure you have Python 3.x installed.

```bash
pip install -r requirements.txt
```

3. **Add the Trained Model**

Place your trained model file as `model.keras` in the project root directory.

4. **Run the Flask App**

```bash
python main.py
```

5. **Access the Web Interface**

Go to: `http://127.0.0.1:5000/`

---

## 🧠 Model Details

- Input Shape: `(224, 224, 3)`
- Model Type: CNN
- Framework: Keras (TensorFlow backend)
- Trained On: Brain MRI Dataset (Tumor vs Normal)

---

## 📊 Dataset

You can use this public dataset:
- [Brain MRI Images for Brain Tumor Detection (Kaggle)](https://www.kaggle.com/datasets/tombackert/brain-tumor-mri-data)

---

## 🛠 Technologies Used

- Python
- Flask
- TensorFlow / Keras
- OpenCV
- HTML & CSS

---

## 📌 Requirements

Listed in `requirements.txt`:

```text
Flask
tensorflow
keras
numpy
opencv-python
```
