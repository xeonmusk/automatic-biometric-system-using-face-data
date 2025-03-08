#-------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------#
#-----------------------------------------------imports-------------------------------------#

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, send_file
import os
import shutil
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from werkzeug.utils import secure_filename
from threading import Thread
import pickle
import urllib.request
from tqdm import tqdm 
from datetime import datetime
from openpyxl import Workbook
import glob
from faker import Faker

#--------------------------------------------------------------------------------------#
#-----------------------APP CONFIGURATION---------------------------------------#
#---------------------------------------------------------------------------------------#

app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['DATASET_UPLOAD_FOLDER'] = '/workspaces/automatic-biometric-system-using-face-data/datasets/actual_data'
app.config['OUTPUT_RESULT_FOLDER'] = '/workspaces/automatic-biometric-system-using-face-data/datasets/preprocessed_data'
app.config['TRAIN'] = '/workspaces/automatic-biometric-system-using-face-data/datasets/preprocessed_data/train'
app.config['VAL'] = '/workspaces/automatic-biometric-system-using-face-data/datasets/preprocessed_data/val'
app.config['TEST'] = '/workspaces/automatic-biometric-system-using-face-data/datasets/preprocessed_data/test'
app.config['IMAGE_UPLOAD_FOLDER'] = '/workspaces/automatic-biometric-system-using-face-data/datasets/testing_data/s'
app.config['MULTIPLE_IMAGE_UPLOAD'] = '/workspaces/automatic-biometric-system-using-face-data/datasets/testing_data/f'
app.config['MODEL_PATH'] = '/workspaces/automatic-biometric-system-using-face-data/models'
app.config['LATEST_MODEL'] = '/workspaces/automatic-biometric-system-using-face-data/latest_model'
app.config['DETECTED_OBJECTS'] = '/workspaces/automatic-biometric-system-using-face-data/Detected_Objects'
app.config['RESULT_FOLDER'] = '/workspaces/automatic-biometric-system-using-face-data/result'
app.config['UPLOADS_FOLDER'] = '/workspaces/automatic-biometric-system-using-face-data/uploads'
app.config['DATAP_FOLDER'] = '/workspaces/automatic-biometric-system-using-face-data/datap'

train_dir = app.config['TRAIN']

# --------------------------------Create generators-------------------------------------#
IMG_HEIGHT, IMG_WIDTH = 128, 128
BATCH_SIZE = 16
train_gen = ImageDataGenerator(rescale=1.0/255.0).flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)
IMG_HEIGHT = 224
IMG_WIDTH = 224
class_labels = ['class1', 'class2', 'class3']
model_generation_time = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register')
def register():
    return render_template('admin/register.html')

@app.route('/admin')
def admin():
    return render_template('admin/admin.html')

#---------------------------------------------------------------------------------#
#----------------------------MAKE CLEAN FOLDER ---------------------------------------#
#---------------------------------------------------------------------------------#
def make_folder_clean(folder_path):
    if os.path.exists(folder_path):
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)

#---------------------------------------------------------------------------------#
#----------------------------GetTotalmebers & Generate emails---------------------------------------#
#---------------------------------------------------------------------------------#
def generate_members_list():
    dataset_path = app.config['TRAIN']
    if not os.path.exists(dataset_path):
        print(f"Dataset path {dataset_path} does not exist.")
        return []
    members = []
    for label in os.listdir(dataset_path):
        label_path = os.path.join(dataset_path, label)
        if os.path.isdir(label_path):
            members.append(label)
    return members

def generate_emails(names, domain):
    emails = []
    for name in names:
        email = f"{name.lower()}@{domain}"
        emails.append(email)
    return emails

#---------------------------------------------------------------------------------#
#----------------------------Send emails to absentees---------------------------------------#
#---------------------------------------------------------------------------------#
def send_emails_to_absenties(emails):
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    from dotenv import load_dotenv

    load_dotenv()
    
    for email in emails:
        SENDER_EMAIL = os.getenv('SENDER_EMAIL', 'vamsi52753@gmail.com')
        SENDER_PASSWORD = os.getenv('SENDER_PASSWORD', 'fcdj noxz nggq dwyn')
        RECIPIENT_EMAIL = email
        SMTP_SERVER = 'smtp.gmail.com'
        SMTP_PORT = 587

        def send_simple_email():
            msg = MIMEMultipart()
            msg['From'] = SENDER_EMAIL
            msg['To'] = RECIPIENT_EMAIL
            msg['Subject'] = "Simple Test Email"
            body = """
            Hello,

            You have been marked absent for today's class. Please ensure to attend the next class.
            check it.

            Best regards,
            """
            msg.attach(MIMEText(body, 'plain'))

            try:
                with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
                    server.starttls()
                    server.login(SENDER_EMAIL, SENDER_PASSWORD)
                    server.send_message(msg)
                print(f"Email sent successfully to {RECIPIENT_EMAIL}")
            except smtplib.SMTPAuthenticationError:
                print("Authentication failed. Check your email credentials.")
            except Exception as e:
                print(f"Failed to send email: {e}")

        send_simple_email()

#---------------------------------------------------------------------------------#
#----------------------------Get single image---------------------------------------#
#---------------------------------------------------------------------------------#
def get_single_image(folder_path):
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.tiff']
    for ext in image_extensions:
        image_files = glob.glob(os.path.join(folder_path, ext))
        if image_files:
            return image_files[0]
    return None

#---------------------------------------------------------------------------------#
#----------------------------Roll numbers generator---------------------------------------#
#---------------------------------------------------------------------------------#
def generate_students_dict(start_roll=1, end_roll=72, prefix="21a51a42", save_to_file=False, filename="students.txt"):
    fake = Faker()
    students = {}
    for i in range(start_roll, end_roll + 1):
        roll = f"{prefix}{i:02d}"
        name = fake.name()
        students[roll] = name
    
    for roll, name in students.items():
        print(f"{roll}: {name}")
    
    if save_to_file:
        with open(filename, "w") as file:
            for roll, name in students.items():
                file.write(f"{roll}: {name}\n")
        print(f"\nDictionary saved to {filename}")
    
    return students

#---------------------------------------------------------------------------------#
#----------------------------UPLOAD DATASET---------------------------------------#
#---------------------------------------------------------------------------------#
@app.route('/uploadDataset', methods=['GET', 'POST'])
def upload_dataset():
    if request.method == 'POST':
        if 'folder' not in request.files:
            flash("No folder part", "error")
            return redirect(url_for('upload_dataset'))
        files = request.files.getlist('folder')
        if not files:
            flash("No selected files", "error")
            return redirect(url_for('upload_dataset'))
        for file in files:
            filename = file.filename
            upload_path = os.path.join(app.config['DATASET_UPLOAD_FOLDER'], filename)
            os.makedirs(os.path.dirname(upload_path), exist_ok=True)
            file.save(upload_path)
        flash("Folder uploaded successfully!", "success")
        return render_template('admin/upload_success.html')
    return render_template('admin/upload.html')

#-----------------------------------------------------------------------------#
#-----------------------------PREPROCESS DATASET----------------------------------#
#-----------------------------------------------------------------------------#
@app.route('/preprocessDataset')
def preprocess():
    dataset_path = app.config['DATASET_UPLOAD_FOLDER']
    output_path = app.config['OUTPUT_RESULT_FOLDER']
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15
    
    for subset in ['train', 'val', 'test']:
        for label in os.listdir(dataset_path):
            os.makedirs(os.path.join(output_path, subset, label), exist_ok=True)
    
    for label in os.listdir(dataset_path):
        label_path = os.path.join(dataset_path, label)
        images = os.listdir(label_path)
        
        if not images:
            print(f"Warning: Directory '{label_path}' is empty. Skipping...")
            continue
        
        train_images, temp_images = train_test_split(images, test_size=(1 - train_ratio), random_state=42)
        val_images, test_images = train_test_split(temp_images, test_size=(test_ratio / (val_ratio + test_ratio)), random_state=42)
        
        for image in train_images:
            shutil.copy(os.path.join(label_path, image), os.path.join(output_path, 'train', label, image))
        for image in val_images:
            shutil.copy(os.path.join(label_path, image), os.path.join(output_path, 'val', label, image))
        for image in test_images:
            shutil.copy(os.path.join(label_path, image), os.path.join(output_path, 'test', label, image))
    return render_template('admin/split_success.html')

#------------------------------------------------------------------------------------------------------------------------#
@app.route('/select_class', methods=['GET', 'POST'])
def select_class():
    models_path = app.config['MODEL_PATH']
    model_files = [f for f in os.listdir(models_path) if f.endswith('.h5')]
    return render_template('user/select_class.html', model_files=model_files)

#-----------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------CREATE AND TRAIN MODEL--------------------------------------------------------#
#-----------------------------------------------------------------------------------------------------------------------#
@app.route('/trainModel', methods=['GET', 'POST'])
def train_model():
    print("Train model POST request received")
    train_dir = app.config['TRAIN']
    val_dir = app.config['VAL']
    test_dir = app.config['TEST']
    
    IMG_HEIGHT, IMG_WIDTH = 128, 128
    BATCH_SIZE = 16
    
    train_gen = ImageDataGenerator(rescale=1.0/255.0).flow_from_directory(
        train_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    val_gen = ImageDataGenerator(rescale=1.0/255.0).flow_from_directory(
        val_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    test_gen = ImageDataGenerator(rescale=1.0/255.0).flow_from_directory(
        test_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    base_model.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(len(train_gen.class_indices), activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    EPOCHS = 20
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        steps_per_epoch=len(train_gen),
        validation_steps=len(val_gen)
    )
    
    test_loss, test_accuracy = model.evaluate(test_gen)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.close()
    
    flash(f"Model trained successfully with Test Accuracy: {test_accuracy * 100:.2f}%")
    
    model_name = request.form.get('model_name', 'model')
    model_filename = f"{model_name}.h5"
    model_path = os.path.join(app.config['MODEL_PATH'], model_filename)
    model.save(model_path)
    
    latest_path = app.config['LATEST_MODEL']
    make_folder_clean(latest_path)
    lat_path = os.path.join(latest_path, model_filename)
    model.save(lat_path)
    print(f"Model saved to {model_path}")
    
    return redirect(url_for('admin'))

#-----------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------UPLOAD SINGLE IMAGE--------------------------------------------------------#
#-----------------------------------------------------------------------------------------------------------------------#
@app.route('/predictSingleImage', methods=['GET', 'POST'])
def predict_single_image():
    make_folder_clean(app.config['IMAGE_UPLOAD_FOLDER'])
    
    if request.method == 'POST':
        if 'image_file' not in request.files:
            flash('No file part', 'error')
            return redirect(request.url)
        
        file = request.files['image_file']
        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)
        
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['IMAGE_UPLOAD_FOLDER'], filename)
            os.makedirs(app.config['IMAGE_UPLOAD_FOLDER'], exist_ok=True)
            file.save(filepath)
            
            flash('Single image uploaded and saved successfully!', 'success')
            return render_template('admin/predict_success.html', message='Single image uploaded and saved.')
    
    return render_template('admin/predict.html')

#------------------------------------------------------------------------------------#
#---------------------------------TEST SINGLE IMAGE-------------------------------#
#------------------------------------------------------------------------------------#
@app.route('/testSingleImage')
def test_single_image():
    try:
        image_path = get_single_image(app.config['IMAGE_UPLOAD_FOLDER'])
        IMG_HEIGHT, IMG_WIDTH = 128, 128
        image = load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
        image_array = img_to_array(image)
        image_array = image_array / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        
        plt.imshow(load_img(image_path))
        plt.axis('off')
        plt.title("Input Image")
        static_dir = os.path.join(os.path.dirname(__file__), 'static')
        os.makedirs(static_dir, exist_ok=True)
        plt.savefig(os.path.join(static_dir, 'input_image.png'))
        plt.close()
        
        latest_model_folder = app.config['LATEST_MODEL']
        files = os.listdir(latest_model_folder)
        if files:
            model_path = os.path.join(latest_model_folder, files[0])
        else:
            raise FileNotFoundError("No models found in the 'LATEST_MODEL' folder")
        
        model = tf.keras.models.load_model(model_path)
        predictions = model.predict(image_array)
        predicted_class = np.argmax(predictions, axis=1)
        
        train_gen = ImageDataGenerator().flow_from_directory(app.config['TRAIN'])
        class_labels = {v: k for k, v in train_gen.class_indices.items()}
        predicted_label = class_labels[predicted_class[0]]
        
        return render_template('admin/predict_result.html', image_path='static/input_image.png', predicted_label=predicted_label)
    
    except Exception as e:
        return f"An error occurred: {str(e)}"

#------------------------------------------------------------------------------------------------------------#
#---------------------------------UPLOAD MULTIPLE IMAGES----------------------------------------------------#
#------------------------------------------------------------------------------------------------------------#
@app.route('/predictMultipleImages', methods=['POST'])
def predict_multiple_images():
    make_folder_clean(app.config['MULTIPLE_IMAGE_UPLOAD'])
    if 'images_folder' not in request.files:
        return redirect(request.url)
    files = request.files.getlist('images_folder')
    for file in files:
        if file.filename == '':
            continue
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['MULTIPLE_IMAGE_UPLOAD'], filename))
    return render_template('admin/m_upload_success.html')

#------------------------------------------------------------------------------------------------------------#
#----------------------------------TEST MULTIPLE IMAGES---------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------#
@app.route('/testing_data/<path:filename>')
def testing_data(filename):
    return send_from_directory(app.config['MULTIPLE_IMAGE_UPLOAD'], filename)

@app.route('/testMultipleImages')
def test_multiple_images():
    try:
        test_images_dir = app.config['MULTIPLE_IMAGE_UPLOAD']
        batch_images, batch_image_names = [], []
        
        for image_name in os.listdir(test_images_dir):
            image_path = os.path.join(test_images_dir, image_name)
            if os.path.isfile(image_path):
                image = load_img(image_path, target_size=(128, 128))
                image_array = img_to_array(image) / 255.0
                batch_images.append(image_array)
                batch_image_names.append(image_name)
        
        if not batch_images:
            return "No images found in the directory."
        
        batch_images = np.array(batch_images)
        
        latest_model_folder = app.config['LATEST_MODEL']
        model_files = os.listdir(latest_model_folder)
        if model_files:
            model_path = os.path.join(latest_model_folder, model_files[0])
        else:
            raise FileNotFoundError("No models found in the 'LATEST_MODEL' folder.")
        
        model = tf.keras.models.load_model(model_path)
        predictions = model.predict(batch_images)
        predicted_classes = np.argmax(predictions, axis=1)
        
        train_gen = ImageDataGenerator().flow_from_directory(app.config['TRAIN'])
        class_labels = {v: k for k, v in train_gen.class_indices.items()}
        
        results = []
        for i, predicted_class in enumerate(predicted_classes):
            if i < len(batch_image_names):
                results.append({
                    'image_name': batch_image_names[i],
                    'image_path': batch_image_names[i],
                    'predicted_class': class_labels[predicted_class]
                })
        
        print("Results:", results)
        return render_template('admin/multiple_predict_result.html', results=results)
    
    except Exception as e:
        return f"An error occurred: {str(e)}"

#------------------------------------------------------------------------------------------------------------#
#-----------------------------------------------FRAMES FROM VIDEO---------------------------------------------#
#------------------------------------------------------------------------------------------------------------#
import cv2

def get_frames(path):
    webCam = cv2.VideoCapture(path)
    currentframe = 0
    
    datap_folder = app.config['DATAP_FOLDER']
    if not os.path.exists(datap_folder):
        os.makedirs(datap_folder)
    
    while True:
        success, frame = webCam.read()
        if not success:
            break
        
        cv2.imshow('Frame', frame)
        cv2.imwrite(os.path.join(datap_folder, f'frame_{currentframe}.jpg'), frame)
        currentframe += 1
        
        if cv2.waitKey(1) & 0xFF == ord('q') or currentframe >= 10:
            break
    
    webCam.release()
    cv2.destroyAllWindows()
    return currentframe

@app.route('/extract-frames')
def extract_frames():
    # Assuming a video file exists in UPLOADS_FOLDER
    video_path = os.path.join(app.config['UPLOADS_FOLDER'], 'recorded_video.mp4')
    frames_count = get_frames(video_path)
    return f"Extracted {frames_count} frames to datap folder"

#------------------------------------------------------------------------------------------------------------#
#-----------------------------------------------YOLO DETECTOR-------------------------------------------------#
#------------------------------------------------------------------------------------------------------------#
@app.route('/detect-objects')
def detect_objects():
    import cv2
    import numpy as np
    import random
    
    def download_file(url, filename):
        try:
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, filename)
            print(f"Downloaded {filename}")
        except Exception as e:
            print(f"Error downloading {filename}: {e}")
            return False
        return True
    
    files = {
        "yolov4.cfg": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg",
        "yolov4.weights": "https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4.weights",
        "coco.names": "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"
    }
    
    for filename, url in files.items():
        if not os.path.exists(filename):
            if not download_file(url, filename):
                raise Exception(f"Failed to download {filename}")
    
    output_folder = app.config['DETECTED_OBJECTS']
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
    
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    
    frame_path = os.path.join(app.config['DATAP_FOLDER'], 'frame_0.jpg')
    img = cv2.imread(frame_path)
    if img is None:
        raise FileNotFoundError("Image file not found. Please check the path.")
    
    original_img = img.copy()
    height, width = img.shape[:2]
    
    blob = cv2.dnn.blobFromImage(
        img,
        scalefactor=1/255.0,
        size=(416, 416),
        swapRB=True,
        crop=False
    )
    
    net.setInput(blob)
    outs = net.forward(output_layers)
    
    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.3
    nms_threshold = 0.4
    
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > conf_threshold:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                x = max(0, int(center_x - w / 2))
                y = max(0, int(center_y - h / 2))
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    if len(indexes) > 0:
        indexes = indexes.flatten()
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    detected_count = 0
    
    for i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = confidences[i]
        color = (0, 255, 0)
        
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        w = min(w, width - x)
        h = min(h, height - y)
        
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, f'{label} {confidence:.2f}', (x, y - 10), font, 0.5, color, 2)
        
        detected_count += 1
        try:
            cropped_img = original_img[y:y+h, x:x+w]
            if cropped_img.size > 0:
                num1 = random.randint(1, 100)
                num2 = random.uniform(1.0, 100.0)
                object_image_path = os.path.join(output_folder, f"{label}_{detected_count}_{num1}_{num2}.jpg")
                cv2.imwrite(object_image_path, cropped_img)
                print(f"Saved {label} at {object_image_path}")
        except Exception as e:
            print(f"Error saving cropped image: {e}")
    
    print(f"Total objects detected: {detected_count}")
    return "Object detection completed"

#-----------------------------------------------CAMERA ACCESS-------------------------------------------------#
def camera_Access():
    cam_folder = os.path.join(os.getcwd(), 'CAM')
    os.makedirs(cam_folder, exist_ok=True)
    url = "http://192.168.1.8:4747/video"
    cap = cv2.VideoCapture(url)
    
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return False
    
    image_counter = 0
    saved_images = []
    is_capturing = False
    
    cv2.namedWindow("DroidCam Feed")
    
    def start_capture(event):
        nonlocal is_capturing
        is_capturing = True
        print("Started capturing")
    
    def stop_capture(event):
        nonlocal is_capturing
        is_capturing = False
        print("Stopped capturing")
    
    cv2.createButton("Start", start_capture, None, cv2.QT_PUSH_BUTTON, 0)
    cv2.createButton("Stop", stop_capture, None, cv2.QT_PUSH_BUTTON, 0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
        status = "Recording" if is_capturing else "Stopped"
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 0, 255) if is_capturing else (255, 255, 255), 2)
        
        cv2.imshow("DroidCam Feed", frame)
        
        if is_capturing:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_name = f"captured_image_{timestamp}_{image_counter}.jpg"
            image_path = os.path.join(cam_folder, image_name)
            
            if cv2.imwrite(image_path, frame):
                print(f"Image saved as {image_path}")
                saved_images.append(image_path)
                image_counter += 1
        
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()
    return saved_images

@app.route('/camera')
def camera():
    saved_images = camera_Access()
    return f"Images saved: {saved_images}"

#------------------------------------------------------------------------------------#
#---------------------------------VIDEO ACCESS--------------------------------------#
#------------------------------------------------------------------------------------#
@app.route('/video_record')
def video_Access():
    import time
    
    url = "http://192.168.1.8:4747/video"
    cap = cv2.VideoCapture(url)
    
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return "Error: Could not open video stream."
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    output_file = os.path.join(app.config['UPLOADS_FOLDER'], "recorded_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, 20.0, (frame_width, frame_height))
    
    start_time = time.time()
    while (time.time() - start_time) < 5:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
        out.write(frame)
        cv2.imshow("Recording...", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"Video saved as {output_file}")
    return render_template('user/camera_status.html')

#------------------------------------------------------------------------------------------------------------#
#-----------------------------------------------UPLOAD VIDEO MANUALLY-------------------------------------------------#
#------------------------------------------------------------------------------------------------------------#
@app.route('/upload', methods=['POST'])
def upload_video():
    UPLOAD_FOLDER = app.config['UPLOADS_FOLDER']
    
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    
    if 'video' not in request.files:
        return "No file part in the request", 400
    
    file = request.files['video']
    
    if file.filename == '':
        return "No selected file", 400
    
    if file:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        return f"File uploaded successfully: {file.filename}", 200

#------------------------------------------------------------------------------------------------------------#
#-----------------------------------------------GET ATTENDANCE VIA IMAGE-------------------------------------------------#
#------------------------------------------------------------------------------------------------------------#
def generate_file(path_of_detected_objects, class_name):
    IMG_HEIGHT, IMG_WIDTH = 128, 128
    BATCH_SIZE = 16
    
    train_dir = app.config['TRAIN']
    train_datagen = ImageDataGenerator(rescale=1./255)
    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    
    class_labels = list(train_gen.class_indices.keys())
    print("Class labels:", class_labels)
    
    model_path = os.path.join(app.config['MODEL_PATH'], f"{class_name}")
    try:
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully")
    except Exception as e:
        raise Exception(f"Error loading model: {str(e)}")
    
    test_images_dir = path_of_detected_objects
    if not os.path.exists(test_images_dir):
        raise FileNotFoundError(f"Directory {test_images_dir} not found")
    
    def load_and_preprocess_images(image_dir):
        images = []
        filenames = []
        for image_name in os.listdir(image_dir):
            if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    image_path = os.path.join(image_dir, image_name)
                    image = load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
                    image_array = img_to_array(image) / 255.0
                    images.append(image_array)
                    filenames.append(image_name)
                except Exception as e:
                    print(f"Error processing {image_name}: {str(e)}")
        return np.array(images), filenames
    
    batch_images, batch_image_names = load_and_preprocess_images(test_images_dir)
    print(batch_images, batch_image_names)
    
    if not batch_images.size:
        raise ValueError("No valid images found in the directory.")
    
    predictions = model.predict(batch_images)
    predicted_classes = np.argmax(predictions, axis=1)
    confidence_scores = np.max(predictions, axis=1)
    
    class_labels = {v: k for k, v in train_gen.class_indices.items()}
    
    results = []
    unique_labels = set()
    i = 0
    for image_name, predicted_class, confidence_score in zip(batch_image_names, predicted_classes, confidence_scores):
        predicted_label = class_labels.get(predicted_class, "Unknown")
        if predicted_label not in unique_labels:
            unique_labels.add(predicted_label)
            confidence_percentage = round(confidence_score * 100, 2)
            i += 1
            results.append({
                "Predicted Label": predicted_label,
                "Accuracy (%)": confidence_percentage,
                "image_path": image_name
            })
    
    presents = list(unique_labels)
    actual = generate_members_list()
    emails = generate_emails(actual, domain="adityatekkali.edu.in")
    print("Emails of Actual members:", emails)
    absents = [x for x in actual if x not in presents]
    emails = generate_emails(absents, domain="adityatekkali.edu.in")
    print("Emails of absentees:", emails)
    # send_emails_to_absenties(emails)
    print("Emails sent to absentees")
    
    results_df = pd.DataFrame(results)
    excel_file_path = os.path.join(app.config['RESULT_FOLDER'], "Unique_Predicted_Results.xlsx")
    os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)
    results_df.to_excel(excel_file_path, index=False)
    print(f"Unique results saved to {excel_file_path}")

@app.route('/get_attendance_video')
def get_attendance_image():
    video_path = os.path.join(app.config['UPLOADS_FOLDER'], 'recorded_video.mp4')
    extract_frames(video_path)
    for i in range(10):
        frame_path = os.path.join(app.config['DATAP_FOLDER'], f'frame_{i}.jpg')
        if os.path.exists(frame_path):
            detect_objects()
    
    if request.args.get('class_name'):
        class_name = request.args.get('class_name')
        generate_file(app.config['DETECTED_OBJECTS'], class_name)
        return display_excel()
    else:
        models_path = app.config['MODEL_PATH']
        model_files = [f.replace('.h5', '') for f in os.listdir(models_path) if f.endswith('.h5')]
        return render_template('user/select_class.html', model_files=model_files)

#------------------------------------------------------------------------------------------------------------#
#-----------------------------------------------DISPLAY EXCEL-------------------------------------------------#
#------------------------------------------------------------------------------------------------------------#
app.config['STATIC_FOLDER'] = 'static'

@app.route('/display_excel')
def display_excel():
    excel_file_path = os.path.join(app.config['RESULT_FOLDER'], "Unique_Predicted_Results.xlsx")
    
    if not os.path.exists(excel_file_path):
        return "Error: Excel file not found. Please check the file path."
    
    df = pd.read_excel(excel_file_path)
    
    static_image_dir = os.path.join(app.static_folder, 'detected_objects')
    if not os.path.exists(static_image_dir):
        os.makedirs(static_image_dir)
    
    detected_objects_dir = app.config['DETECTED_OBJECTS']
    for img_file in os.listdir(detected_objects_dir):
        if img_file.endswith(('.jpg', '.png', '.jpeg')):
            src_path = os.path.join(detected_objects_dir, img_file)
            dest_path = os.path.join(static_image_dir, img_file)
            if not os.path.exists(dest_path):
                shutil.copy(src_path, dest_path)
    
    df['image_path'] = df['image_path'].apply(
        lambda x: f'<img src="/static/detected_objects/{x}" alt="Image" style="max-width: 100px; height: auto;">'
    )
    
    table_html = df.to_html(classes='table table-striped', index=False, escape=False)
    return render_template('user/display_excel.html', table=table_html)

@app.route('/download_excel')
def download_excel():
    excel_file_path = os.path.join(app.config['RESULT_FOLDER'], "Unique_Predicted_Results.xlsx")
    
    if not os.path.exists(excel_file_path):
        return "Error: Excel file not found. Please check the file path."
    
    try:
        return send_file(
            excel_file_path,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name='attendance_report.xlsx'
        )
    except Exception as e:
        return str(e)

#------------------------------------------------------------------------------------------------------------#
#-----------------------------------------------DISPLAY AVAILABLE MODELS-------------------------------------------------#
#------------------------------------------------------------------------------------------------------------#
@app.route('/available_models')
def available_models():
    try:
        models_path = app.config['MODEL_PATH']
        model_files = [f for f in os.listdir(models_path) if f.endswith('.h5')]
        model_files = [os.path.splitext(f)[0] for f in model_files]
        return render_template('user/models_list.html', models=model_files)
    except Exception as e:
        return f"Error accessing models: {str(e)}"

#------------------------------------------------------------------------------------------------------------#
#-----------------------------------------------RUN FLASK-------------------------------------------------#
#------------------------------------------------------------------------------------------------------------#
if __name__ == '__main__':
    app.run(debug=True)
