🔐 Automatic Biometric Attendance System Using Face Data

GitHub: github.com/xeonmusk/automatic-biometric-system-using-face-data

A complete end-to-end facial recognition-based attendance system using deep learning and real-time computer vision. It enables institutions to automate attendance with high accuracy using live camera input and pre-trained models.


🧠 Core Features & Workflow:

1. 📂 Dataset Upload & Organization

Users upload labeled face images.

Images are auto-organized into person-specific folders under dataset/.



2. 🔖 Image Labeling & Annotation

Each folder name is treated as the class label.

All images are automatically annotated and label-encoded for model training.



3. 🧼 Image Preprocessing

Converts all images to grayscale, resizes to standard dimensions, and stores in processed_images/.



4. 📦 Custom Dataset Creation

Preprocessed images + label encodings → used to create training-ready datasets.



5. 📐 Facial Embedding Generation

Extracts 128D facial encodings via face_recognition library.

Saves encodings and labels in .pkl files inside encodings/.



6. 🎓 Model Training (Per-Person)

Creates and trains a separate MobileNetV2 model for each individual using their labeled dataset.

Trained models are saved as .h5 files in the models/ directory.



7. 📸 Single & Group Image Prediction

Allows user to upload single or multi-face images.

Predicts identities and marks present students.



8. 📷 Real-Time Attendance via Live Camera ✅ (New)

Uses webcam to capture live video feed.

Randomly samples frames during the session.

Each frame is passed through YOLO (You Only Look Once) object detection to detect faces quickly and accurately.

Detected face regions are cropped and passed to the trained recognition model.

Matches known identities and records attendance.



9. 🕒 Timed Attendance Logging ✅ (New)

Attendance is marked for students only when the user clicks “Get Attendance”.

Ensures that attendance is tied to a specific time window or session.



10. 🚫 Absentee Detection



Compares the set of known students with those marked present.

Identifies absentees in real-time.


11. 📧 Email Notification



Sends absentee lists to configured email addresses via SMTP.



---

⚙️ Tech Stack

Component	Technology

Backend	Flask (Python)
Model Architecture	MobileNetV2 (TensorFlow/Keras)
Face Detection	YOLO (You Only Look Once)
Face Recognition	face_recognition, OpenCV
Label Encoding	sklearn, custom logic
Email Notification	smtplib, email.message
UI	HTML, Jinja2 (Flask templates)



---

📁 Project Folder Structure

├── dataset/              # Original labeled images
├── processed_images/     # Grayscale, resized images
├── encodings/            # .pkl facial embeddings
├── models/               # Trained .h5 models per person
├── static/               # Uploaded prediction images
├── templates/            # HTML frontend
├── app.py                # Main Flask application


---

✅ Key Highlights

🔍 Uses YOLO for real-time face detection from live webcam feed.

🧠 Trains custom per-person MobileNetV2 models for better accuracy.

🏷️ Fully automated dataset labeling, annotation, and training pipeline.

⏱️ Attendance marking tied to specific time clicks for accountability.

📧 Automatic absentee alerts via email.

⚙️ Modular, clean structure for easy deployment in educational or office environments.
