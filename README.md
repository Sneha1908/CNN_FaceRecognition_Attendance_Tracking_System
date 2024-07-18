FACE RECOGNITION SYSTEM FOR AUTOMATED ATTENDANCE TRACKING

This project implements a face recognition-based attendance system using Convolutional Neural Networks (CNNs). The system captures images of individuals, processes them, trains a CNN model for face recognition, and records attendance in real-time. Additionally, it provides a web interface to view attendance records using Streamlit.

Introduction:

This project is designed to automate the attendance system using face recognition technology. The system captures images of individuals, processes them, and trains a Convolutional Neural Network (CNN) to recognize faces. Once the model is trained, it can be used to mark attendance in real-time by recognizing faces from a webcam feed.

Features:

Capture and save images of individuals for training.
Preprocess images and train a CNN model for face recognition.
Real-time face detection and recognition using a webcam.
Record attendance in CSV files.
Display attendance records using Streamlit.

Installation Prerequisites:

Python 3.6 or higher
OpenCV
TensorFlow/Keras
Streamlit
NumPy
Matplotlib
Scikit-learn
Pandas

Project Structure:

face-recognition-attendance-system/
│
├── models/
│   ├── cnn_face_recognizer_best_model.h5
│   └── label_map.pkl
│
├── data/
│   └── (Captured images of individuals organized by folders named with ID-Name format)
│
├── Attendance/
│   └── (CSV files with attendance records)
│
├── capture_images.py
├── train_model.py
├── mark_attendance.py
├── view_attendance.py
├── requirements.txt
└── README.md

Model Training:

The script train_model.py is used to train the CNN model. It includes functions for loading and preprocessing images, defining the CNN architecture, training the model, and evaluating its performance. The model and label map are saved for later use in real-time attendance marking.

Real-Time Attendance:

The script mark_attendance.py uses the trained CNN model to recognize faces in real-time from a webcam feed. It records attendance in a CSV file, including the ID, name, and timestamp.

Attendance Record Viewing:

The script view_attendance.py provides a Streamlit web interface to view the attendance records for the current date. It highlights the maximum values for better visualization.

Results:
The results of the face recognition model, including accuracy, F1 score, precision, recall, and a classification report, are displayed after training the model. The system's performance can be visualized using plots of training and validation accuracy and loss.


