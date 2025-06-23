# 🎓 Face Recognition System for Automated Attendance Tracking

An end-to-end **CNN-based facial recognition system** for real-time attendance tracking. This project combines computer vision, deep learning, and an interactive web interface to deliver a smart, accurate, and user-friendly solution for automating attendance in educational and professional settings.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Technologies Used](#-technologies-used)
- [Installation & Setup](#-installation--setup)
- [How to Use](#-how-to-use)
- [Screenshots](#-screenshots)
- [Evaluation Metrics](#-evaluation-metrics)
- [Future Improvements](#-future-improvements)
- [License](#-license)
- [Author](#-author)
---


## 🧠 Overview

Traditional attendance systems are manual, time-consuming, and prone to errors. This project presents a smart alternative by leveraging computer vision and deep learning to automate attendance tracking using face recognition.

The system uses a webcam to capture live video, detects and recognizes registered faces using a convolutional neural network (CNN), and logs attendance in real time. A Streamlit-based web dashboard allows administrators to view, search, and export attendance records easily.

Designed for scalability and practical use, the system ensures accurate recognition across multiple users and varying environmental conditions.

---

## ✨ Key Features

### 🔍 Real-Time Face Detection
- Captures live video using webcam.
- Uses Haar Cascade Classifier (OpenCV) for real-time face localization.
- Detects multiple faces simultaneously in varying lighting conditions.

### 🧠 CNN-Based Face Recognition
- Custom Convolutional Neural Network (CNN) built using TensorFlow and Keras.
- Trained on grayscale face images resized to 128x128.
- Model architecture includes Conv2D, MaxPooling2D, BatchNormalization, Dropout, and Dense layers.
- Achieves >94% accuracy with data augmentation techniques.

### 👤 Dynamic User Registration
- New users can register through CLI by entering their ID and Name.
- Captures up to 100 face images per user and stores them in a labeled directory.
- Existing user images can be appended to improve model robustness.

### 🏷️ Label Management & Encoding
- Encodes user identities as numeric class labels using one-hot encoding.
- Stores label mapping as a pickle file (`label_map.pkl`) for use during recognition.

### ✅ Automated Attendance Logging
- Recognized faces are logged into a timestamped CSV file inside `/Attendance/`.
- Each row includes ID, Name, and exact recognition time.
- Duplicate entries per session are avoided using a confidence threshold and tracking dictionary.

### 📊 Performance Evaluation
- Accuracy, Precision, Recall, and F1 Score calculated using Scikit-learn.
- Real-time graphs plotted using Matplotlib to visualize training/validation performance.

### 🧾 Streamlit Attendance Dashboard
- Displays real-time attendance logs in a web UI.
- Supports date selection, search by ID/name, and table display.
- Exports filtered logs to CSV and auto-generates a styled PDF using ReportLab.

### 🧪 Modular Development
- `train_model.ipynb` for model training.
- `test_model.ipynb` for running live recognition.
- `attendance_app.py` (Streamlit) for report viewing/exporting.
- Follows a clean, modular structure separating logic, UI, and ML code.

---

## 🏗️ System Architecture

This project follows a modular architecture integrating face detection, CNN-based recognition, attendance logging, and a web-based reporting interface.

### 🧩 Component Flow:

User → Camera → Face Detection → CNN Model → Recognition → Attendance Logging → Streamlit Dashboard

### 🧱 Architecture Layers

1. **Input Layer**
   - Captures real-time video using a connected webcam with OpenCV.

2. **Data Preprocessing Module**
   - Converts video frames to grayscale.
   - Detects face regions and resizes them to 128x128 pixels.
   - Normalizes pixel values to [0, 1] range.
   - Applies data augmentation (rotation, flipping, zooming, shifting) using Keras' `ImageDataGenerator`.

3. **Model Training**
   - A convolutional neural network (CNN) is built using the Keras Sequential API.
   - Includes layers such as Conv2D, MaxPooling2D, BatchNormalization, Dropout, and Dense.
   - Trained using categorical cross-entropy loss and the Adam optimizer.
   - Model performance is evaluated using training and validation accuracy/loss plots.

4. **Recognition & Prediction**
   - Loads the trained `.h5` model and label mapping (`.pkl`).
   - Detects faces from live webcam feed.
   - Predicts identity using the CNN model and logs only confident matches.
   - Avoids duplicate entries by tracking detection counts per user.

5. **Attendance Logging**
   - For each recognized face, the system logs attendance into a CSV file named `Attendance_<date>.csv`.
   - Each log includes the user ID, name, and timestamp.
   - Attendance files are organized in a local folder.

6. **Web Dashboard (Streamlit)**
   - Displays attendance data in a tabular format.
   - Supports filtering by date, ID, or name.
   - Allows exporting data to CSV or generating downloadable PDF reports.

---

## 🛠 Technologies Used

| Category             | Tools / Libraries                                     |
|----------------------|--------------------------------------------------------|
| Deep Learning Model  | Convolutional Neural Network (CNN) via Keras, TensorFlow |
| Computer Vision      | OpenCV, Haar Cascade Classifier                        |
| Data Augmentation    | Keras `ImageDataGenerator`                             |
| Data Handling        | NumPy, Pandas                                          |
| Model Evaluation     | Scikit-learn (Accuracy, Precision, Recall, F1)         |
| Visualization        | Matplotlib                                             |
| Data Export          | CSV (built-in), ReportLab (PDF generation)             |
| User Interface       | Streamlit (Web Dashboard), Command-Line Interface      |
| Storage & Persistence| Pickle (label maps), Local directory structure         |

## 🔧 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Sneha1908/CNN_FaceRecognition_Attendance_Tracking_System.git
cd CNN_FaceRecognition_Attendance_Tracking_System
```
### 2. Create a Virtual Environment (optional but recommended)
```bash
python -m venv venv
venv\Scripts\activate
```
### 3. Install Dependencies

Make sure your virtual environment is activated, then install all required packages using:

```bash
pip install -r requirements.txt
```
### 4. Train the CNN Model

Open `train_model.ipynb` in your preferred Python environment (VS Code or Jupyter Notebook or Google Colab).
```bash
jupyter notebook train_model.ipynb
```

The model is trained on a combination of:

- Real-time facial images captured through webcam during user registration (stored in user-specific folders)
- External celebrity face dataset for additional diversity and better generalization

Then run all the cells to:

- Load and preprocess the dataset (resizing, normalization, augmentation)
- Train the CNN model using Keras with TensorFlow backend
- Save the trained model as `cnn_face_recognizer_best_model.h5`
- Save the label mapping as `label_map.pkl`

### 5. Run Real-Time Face Recognition & Attendance Logging

Open the recognition notebook using:

```bash
jupyter notebook test_model.ipynb
```
Run all the cells to:

- Access your system's webcam
- Detect faces using Haar Cascade Classifier
- Recognize faces using the trained CNN model
- Log attendance (ID, Name, Timestamp) for each recognized person
- Automatically create or append a CSV file in the `Attendance/` folder (e.g., `Attendance_2025-06-24.csv`)

### 6. Launch the Streamlit Dashboard

Run the following command in your terminal:

```bash
streamlit run attendance_app.py
```
This will:

- Launch a web-based dashboard in your browser
- Let you view attendance logs by selecting a date
- Support searching by ID or name
- Allow exporting attendance data to:
  - CSV file
  - PDF report (auto-generated using ReportLab)












