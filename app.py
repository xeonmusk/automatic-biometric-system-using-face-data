#-------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------#
#-----------------------------------------------imports-------------------------------------#

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, send_file# For flask
import os# For file paths
import shutil# For moving files
from sklearn.model_selection import train_test_split# For splitting data
import tensorflow as tf# For deep learning
from keras.applications.mobilenet_v2 import MobileNetV2# For MobileNetV2 model
from tensorflow.keras.models import Model, load_model# For model creation and loading
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D# For additional layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array# For image processing
import matplotlib.pyplot as plt# For plotting
import numpy as np# For numerical operations
import pandas as pd# For data manipulation
from werkzeug.utils import secure_filename# For secure file name
from threading import Thread# For running the download in a separate thread
import pickle# For saving the model
import cv2# For image processing
import urllib.request# For downloading files
from tqdm import tqdm # tqdm is a progress bar library
from datetime import datetime# For date and time
from openpyxl import Workbook# For Excel
import glob# For file search
from faker import Faker# For generating random names
#--------------------------------------------------------------------------------------#
#-----------------------ALL PATHS OF DIRECTORIES---------------------------------------#
#---------------------------------------------------------------------------------------#

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Add a secret key for flash messages
app.config['DATASET_UPLOAD_FOLDER'] = r'C:\Users\Lenovo\Desktop\final_project-main\datasets\actual_data'
app.config['OUTPUT_RESULT_FOLDER'] = r'C:\Users\Lenovo\Desktop\final_project-main\datasets\preprocessed_data'  # Path to save the split dataset
app.config['TRAIN'] = r'C:\Users\Lenovo\Desktop\final_project-main\datasets\preprocessed_data\train'
app.config['VAL'] = r'C:\Users\Lenovo\Desktop\final_project-main\datasets\preprocessed_data\val'
app.config['TEST'] = r'C:\Users\Lenovo\Desktop\final_project-main\datasets\preprocessed_data\test'
app.config['IMAGE_UPLOAD_FOLDER'] = r'C:\Users\Lenovo\Desktop\final_project-main\datasets\testing_data\s'
app.config['MULTIPLE_IMAGE_UPLOAD'] = r'C:\Users\Lenovo\Desktop\final_project-main\datasets\testing_data\f'
app.config['MODEL_PATH'] = r'C:\Users\Lenovo\Desktop\final_project-main\models'
app.config['LATEST_MODEL'] = r'C:\Users\Lenovo\Desktop\final_project-main\latest_model'     # Folder to save captured images    
#labels
train_dir = r"C:\Users\Lenovo\Desktop\final_project-main\datasets\preprocessed_data\train"

# --------------------------------Create generators-------------------------------------#
#---------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------#
IMG_HEIGHT, IMG_WIDTH = 128, 128
BATCH_SIZE = 16
train_gen = ImageDataGenerator(rescale=1.0/255.0).flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)
IMG_HEIGHT = 224  # Example height\Users\geeth\Downlo
IMG_WIDTH = 224   # Example width
class_labels = ['class1', 'class2', 'class3']  # Example class labels
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
            # Remove all files and subdirectories
            for item in os.listdir(folder_path):
                item_path = os.path.join(folder_path, item)
                if os.path.isfile(item_path):
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)

#---------------------------------------------------------------------------------#
#----------------------------GetTotalmebers & Genarate emails for them---------------------------------------#
#---------------------------------------------------------------------------------#
def generate_members_list():
    import os
    dataset_path=r"C:\Users\Lenovo\Desktop\final_project-main\datasets\preprocessed_data\train"
    if not os.path.exists(dataset_path):
        print(f"Dataset path {dataset_path} does not exist.")
        return []
    members = []
    # Iterate through each subdirectory (label)
    for label in os.listdir(dataset_path):
        label_path = os.path.join(dataset_path, label)
        if os.path.isdir(label_path):
            members.append(label)
    return members

# Function to generate email addresses
def generate_emails(names,domain):
    emails = []
    for name in names:
        email = f"{name.lower()}@{domain}"
        emails.append(email)
    return emails
# Generate the emails
#---------------------------------------------------------------------------------#
#----------------------------GetTotalmebers & Genarate emails for them---------------------------------------#
#---------------------------------------------------------------------------------#
def send_emails_to_absenties(emails):
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    import os
    from dotenv import load_dotenv

# Load environment variables from a .env file for security (optional)
    load_dotenv()

# Email configuration
    for email in emails:
        SENDER_EMAIL = os.getenv('SENDER_EMAIL', 'vamsi52753@gmail.com')  # Replace with your email
        SENDER_PASSWORD = os.getenv('SENDER_PASSWORD', 'fcdj noxz nggq dwyn')  # Replace with your app-specific password
        RECIPIENT_EMAIL =email # Replace with the recipient's email
        SMTP_SERVER = 'smtp.gmail.com'
        SMTP_PORT = 587  # Use 587 for TLS

# Function to send a simple email
        def send_simple_email():
    # Create the email content
            msg = MIMEMultipart()
            msg['From'] = SENDER_EMAIL
            msg['To'] = RECIPIENT_EMAIL
            msg['Subject'] = "Simple Test Email"

            body = """
            Hello,

            You have been marked absent for today's class. Please ensure to attend the next class.
            check  it.

            Best regards,
            """
            msg.attach(MIMEText(body, 'plain'))

    # Send the email
            try:
                with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
                    server.starttls()  # Enable TLS
                    server.login(SENDER_EMAIL, SENDER_PASSWORD)
                    server.send_message(msg)
                print(f"Email sent successfully to {RECIPIENT_EMAIL}")
            except smtplib.SMTPAuthenticationError:
                print("Authentication failed. Check your email credentials.")
            except Exception as e:
                print(f"Failed to send email: {e}")

#---------------------------------------------------------------------------------#
#----------------------------Get single image---------------------------------------#
#---------------------------------------------------------------------------------#


def get_single_image(folder_path):
    # Supported image file extensions
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.tiff']

    # Search for the first image file in the folder
    for ext in image_extensions:
        image_files = glob.glob(os.path.join(folder_path, ext))
        if image_files:  # If any image file is found
            return image_files[0]  # Return the first (and only) image file

    # If no image file is found
    return None
#---------------------------------------------------------------------------------#
#----------------------------roll numbers  generator---------------------------------------#
#---------------------------------------------------------------------------------#
from faker import Faker

def generate_students_dict(start_roll=1, end_roll=72, prefix="21a51a42", save_to_file=False, filename="students.txt"):
    """
    Generate a dictionary of students with roll numbers and names.

    Parameters:
        start_roll (int): Starting roll number (default: 1).
        end_roll (int): Ending roll number (default: 72).
        prefix (str): Prefix for roll numbers (default: "21a51a42").
        save_to_file (bool): Whether to save the dictionary to a file (default: False).
        filename (str): Name of the file to save the dictionary (default: "students.txt").

    Returns:
        dict: A dictionary with roll numbers as keys and names as values.
    """
    fake = Faker()  # Initialize Faker to generate random names
    students = {}   # Dictionary to store roll numbers and names

    # Generate roll numbers and names
    for i in range(start_roll, end_roll + 1):
        roll = f"{prefix}{i:02d}"  # Format roll number with leading zeros
        name = fake.name()         # Generate a random name
        students[roll] = name

    # Print the dictionary
    for roll, name in students.items():
        print(f"{roll}: {name}")

    # Optionally, save the dictionary to a file
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
    #make_folder_clean(app.config['DATASET_UPLOAD_FOLDER'])
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
    # Ratios for train, validation, and test
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15
    # Create output directories
    for subset in ['train', 'val', 'test']:
        for label in os.listdir(dataset_path):
            os.makedirs(os.path.join(output_path, subset, label), exist_ok=True)

    # Split data
    for label in os.listdir(dataset_path):
        label_path = os.path.join(dataset_path, label)
        images = os.listdir(label_path)

        # Check if the directory contains images
        if not images:
            print(f"Warning: Directory '{label_path}' is empty. Skipping...")
            continue  # Skip to the next label if the directory is empty

        # Split into train and temp (val+test)
        train_images, temp_images = train_test_split(images, test_size=(1 - train_ratio), random_state=42)

        # Split temp into val and test
        val_images, test_images = train_test_split(temp_images, test_size=(test_ratio / (val_ratio + test_ratio)), random_state=42)

        # Copy files to respective folders
        for image in train_images:
            shutil.copy(os.path.join(label_path, image), os.path.join(output_path, 'train', label, image))
        for image in val_images:
            shutil.copy(os.path.join(label_path, image), os.path.join(output_path, 'val', label, image))
        for image in test_images:
            shutil.copy(os.path.join(label_path, image), os.path.join(output_path, 'test', label, image))
    return render_template('admin/split_success.html')
    
#------------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------------------#

@app.route('/select_class', methods=['GET', 'POST'])
def select_class():
    models_path = r'C:\Users\Lenovo\Desktop\final_project-main\models'
    model_files = [f for f in os.listdir(models_path) if f.endswith('.h5')]
    return render_template('user/select_class.html', model_files=model_files)

#-----------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------CREATE AND TRAIN MODEL--------------------------------------------------------#
#-----------------------------------------------------------------------------------------------------------------------#

@app.route('/trainModel', methods=['GET', 'POST'])
def train_model():
        print("Train model POST request received")
        # Paths to train, val, and test datasets
        train_dir=app.config['TRAIN']
        val_dir=app.config['VAL'] 
        test_dir=app.config['TEST']
        # Image dimensions and batch size
        IMG_HEIGHT, IMG_WIDTH = 128, 128
        BATCH_SIZE = 16
        # Create generators
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
        # Load MobileNetV2 with pretrained weights, excluding the top layer
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
        # Freeze the base model's layers
        base_model.trainable = False
        # Add custom layers on top of the base model
        x = base_model.output
        x = GlobalAveragePooling2D()(x)  # Global average pooling
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)  # Regularization
        predictions = Dense(len(train_gen.class_indices), activation='softmax')(x)  # Output layer
        # Create the full model
        model = Model(inputs=base_model.input, outputs=predictions)
        # Compile the model
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        # Train the model
        EPOCHS = 20
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=EPOCHS,
            steps_per_epoch=len(train_gen),
            validation_steps=len(val_gen)
        )
        # Evaluate the model on test data
        test_loss, test_accuracy = model.evaluate(test_gen)
        print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
        # Plot training and validation accuracy
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.legend()
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.close()
        flash(f"Model trained successfully with Test Accuracy: {test_accuracy * 100:.2f}%")
        # Generate unique model name with timestamp
        #model_path = os.path.join(app.config['MODEL_PATH'], name_of_the_model)
        # Generate unique model name with timestamp
        model_name = request.form.get('model_name', 'model')
        model_filename = f"{model_name}.h5"
        model_path = os.path.join(app.config['MODEL_PATH'], model_filename)
        model.save(model_path)
        latest_path=app.config['LATEST_MODEL'] 
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
    import shutil
    folder_path = app.config['IMAGE_UPLOAD_FOLDER']
    if os.path.exists(folder_path):
        # Remove all files and subdirectories
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
    
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
            
            # Add your prediction logic here
            
            flash('Single image uploaded and saved successfully!', 'success')
            return render_template('admin/predict_success.html', message='Single image uploaded and saved.')
    
    return render_template('admin/predict.html')


#------------------------------------------------------------------------------------#
#---------------------------------TEST SINGLE IMAGE-------------------------------#
#------------------------------------------------------------------------------------#

@app.route('/testSingleImage')
def test_single_image():
    try:
        # Get the path of the single image (assuming get_single_image() function exists)
        image_path = get_single_image(app.config['IMAGE_UPLOAD_FOLDER'])
        
        # Set the image input size for the model (adjust based on your model's requirements)
        IMG_HEIGHT, IMG_WIDTH = 128, 128  # Make sure this matches your model's expected input
        image = load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))  # Resize image
        image_array = img_to_array(image)  # Convert image to NumPy array
        image_array = image_array / 255.0  # Normalize pixel values to [0, 1]
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

        # Plot and save the original image to static folder
        plt.imshow(load_img(image_path))
        plt.axis('off')  # Hide axes
        plt.title("Input Image")
        
        # Ensure the static directory exists
        static_dir = os.path.join(os.path.dirname(__file__), 'static')
        os.makedirs(static_dir, exist_ok=True)
        
        # Save the image to the static folder for web display
        plt.savefig(os.path.join(static_dir, 'input_image.png'))
        plt.close()

        # Get the latest model from the folder
        latest_model_folder = app.config['LATEST_MODEL']
        files = os.listdir(latest_model_folder)
        if files:
            model_path = os.path.join(latest_model_folder, files[0])  # Get full model path
        else:
            raise FileNotFoundError("No models found in the 'LATEST_MODEL' folder")

        # Load the model
        model = tf.keras.models.load_model(model_path)

        # Predict the class
        predictions = model.predict(image_array)
        predicted_class = np.argmax(predictions, axis=1)  # Get the index of the highest probability

        # Get class labels from the training data
        train_gen = ImageDataGenerator().flow_from_directory(app.config['TRAIN'])
        class_labels = train_gen.class_indices
        class_labels = {v: k for k, v in class_labels.items()}  # Reverse to map index to class

        predicted_label = class_labels[predicted_class[0]]

        # Return the result template with the image and prediction
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

from flask import send_from_directory

# Add this route to serve files from the upload directory
@app.route('/testing_data/<path:filename>')
def testing_data(filename):
    return send_from_directory(app.config['MULTIPLE_IMAGE_UPLOAD'], filename)

@app.route('/testMultipleImages')
def test_multiple_images():
    try:
        # Get the directory containing the images for prediction
        test_images_dir = app.config['MULTIPLE_IMAGE_UPLOAD']

        batch_images, batch_image_names = [], []

        # Loop through each image in the directory and preprocess it
        for image_name in os.listdir(test_images_dir):
            image_path = os.path.join(test_images_dir, image_name)
            if os.path.isfile(image_path):
                image = load_img(image_path, target_size=(128, 128))  # Resize the image
                image_array = img_to_array(image) / 255.0  # Normalize the image array
                batch_images.append(image_array)
                batch_image_names.append(image_name)

        if not batch_images:
            return "No images found in the directory."

        # Convert the list of images to a NumPy array
        batch_images = np.array(batch_images)

        # Get the latest model from the folder (dynamic loading)
        latest_model_folder = app.config['LATEST_MODEL']
        model_files = os.listdir(latest_model_folder)
        if model_files:
            model_path = os.path.join(latest_model_folder, model_files[0])  # Use the first model found
        else:
            raise FileNotFoundError("No models found in the 'LATEST_MODEL' folder.")

        # Load the model
        model = tf.keras.models.load_model(model_path)

        # Get predictions for the batch of images
        predictions = model.predict(batch_images)
        predicted_classes = np.argmax(predictions, axis=1)  # Get the index of the highest probability

        # Get class labels from the training data
        train_gen = ImageDataGenerator().flow_from_directory(app.config['TRAIN'])
        class_labels = {v: k for k, v in train_gen.class_indices.items()}  # Reverse to map index to class

        results = []
        # Prepare results for each image in the batch
        for i, predicted_class in enumerate(predicted_classes):
            if i < len(batch_image_names):
                results.append({
                    'image_name': batch_image_names[i],  
                    'image_path': batch_image_names[i],  # Use only the filename
                    'predicted_class': class_labels[predicted_class]  
                })

        print("Results:", results)  # Debugging output to the console

        # Return the results in an HTML template
        return render_template('admin/multiple_predict_result.html', results=results)

    except Exception as e:
        return f"An error occurred: {str(e)}"


#------------------------------------------------------------------------------------------------------------#
#-----------------------------------------------FRAMES FROM VIDEO---------------------------------------------#
#------------------------------------------------------------------------------------------------------------#

def get_frames(path):  # Fixed typo in function name
    webCam = cv2.VideoCapture(path)
    currentframe = 0
    
    if not os.path.exists('datap'):
        os.makedirs('datap')

    while True:
        success, frame = webCam.read()
        
        if not success:
            break
            
        # Show frame in window instead of cv2_imshow
        cv2.imshow('Frame', frame)
        
        # Save frame
        cv2.imwrite(f'datap/frame_{currentframe}.jpg', frame)
        currentframe += 1
        
        # Break on 'q' press or after 5 frames
        if cv2.waitKey(1) & 0xFF == ord('q') or currentframe >= 10:
            break
    
    webCam.release()
    cv2.destroyAllWindows()
    return currentframe

@app.route('/extract-frames')
def extract_frames(path):
    frames_count = get_frames(path)
    return f"Extracted {frames_count} frames to datap folder"

#------------------------------------------------------------------------------------------------------------#
#-----------------------------------------------YOLO DETECTOR-------------------------------------------------#
#------------------------------------------------------------------------------------------------------------#

@app.route('/detect-objects')
def detect_objects(frame_path):
    import cv2
    import numpy as np
    import os
    import urllib.request
    import random

    # Download YOLOv4 config, weights, and class names if not already available
    def download_file(url, filename):
        try:
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, filename)
            print(f"Downloaded {filename}")
        except Exception as e:
            print(f"Error downloading {filename}: {e}")
            return False
        return True

    # YOLOv4 files config
    files = {
        "yolov4.cfg": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg",
        "yolov4.weights": "https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4.weights",
        "coco.names": "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"
        }

    # Download files if not present
    for filename, url in files.items():
        if not os.path.exists(filename):
            if not download_file(url, filename):
                raise Exception(f"Failed to download {filename}")
    # Clean the output folder before detecting new objects
    #folder_path = r"C:\Users\Lenovo\Desktop\final_project-main\Detected_Objects"
    #make_folder_clean(folder_path)

    # Create a folder to save images with bounding boxes
    output_folder = r"C:\Users\Lenovo\Desktop\final_project-main\Detected_Objects"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load YOLOv4
    net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")

    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    # Load image
    img = cv2.imread(frame_path)
    if img is None:
        raise FileNotFoundError("Image file not found. Please check the path.")

    # Keep original size for better detection
    original_img = img.copy()
    height, width = img.shape[:2]

    # Improve blob parameters
    blob = cv2.dnn.blobFromImage(
        img,
        scalefactor=1/255.0,
        size=(416, 416),
        swapRB=True,
        crop=False
    )

    # Set network input and get detections
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Detection parameters
    class_ids = []
    confidences = []
    boxes = []

    # Lower confidence threshold for better detection
    conf_threshold = 0.3
    nms_threshold = 0.4

    # Process detections
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > conf_threshold:
                print(f"Detection: Class ID: {class_id}, Confidence: {confidence}")

                # Scale coordinates back to original image size
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Calculate corners
                x = max(0, int(center_x - w / 2))
                y = max(0, int(center_y - h / 2))

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    print(f"Total detections before NMS: {len(boxes)}")

    # Apply non-max suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    # Convert indexes to the correct format
    if len(indexes) > 0:
        indexes = indexes.flatten()

    # Draw boxes
    font = cv2.FONT_HERSHEY_SIMPLEX
    detected_count = 0

    for i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = confidences[i]
        color = (0, 255, 0)  # Green color for boxes

        # Ensure coordinates are within image bounds
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        w = min(w, width - x)
        h = min(h, height - y)

        # Draw rectangle and label
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, f'{label} {confidence:.2f}', (x, y - 10), font, 0.5, color, 2)

        # Save detected object
        detected_count += 1
        try:
            cropped_img = original_img[y:y+h, x:x+w]
            if cropped_img.size > 0:  # Check if crop is valid
                object_image_path = os.path.join(output_folder, f"{label}_{detected_count}_{random.randint(1, 100)+random.uniform(1.0, 100.0)}.jpg")
                cv2.imwrite(object_image_path, cropped_img)
                print(f"Saved {label} at {object_image_path}")
        except Exception as e:
            print(f"Error saving cropped image: {e}")

    # Save the original image with bounding boxes
    #output_image_path = os.path.join(output_folder, "detected_objects.jpg")
    #cv2.imwrite(output_image_path, img)
    #print(f"Saved image with bounding boxes at {output_image_path}")
    cropped_img = img[y:y+h, x:x+w]
    # Generates random numbers between 1 and 100
    num1=random.randint(1, 100)
    num2=random.uniform(1.0, 100.0)

    object_image_path = os.path.join(output_folder, f"object_{detected_count}_{num1}_{num2}.jpg")
    #cv2.imwrite(object_image_path, cropped_img)
    print(f"Saved {label} at {object_image_path}")

    # Display the original image with bounding boxes
    #cv2.imshow("Detected Objects", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f"Total objects detected: {detected_count}")

#-----------------------------------------------CAMERA ACCESS-------------------------------------------------#
#------------------------------------------------------------------------------------------------------------#
#-----------------------------------------------CAMERA ACCESS-------------------------------------
from datetime import datetime
import time
def camera_Access():
    cam_folder = os.path.join(os.getcwd(), 'CAM')
    os.makedirs(cam_folder, exist_ok=True)
    url = "http://192.168.1.27:4747/video"
    cap = cv2.VideoCapture(url)

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return False

    image_counter = 0
    saved_images = []
    is_capturing = False

    # Create window and buttons
    cv2.namedWindow("DroidCam Feed")
    
    def start_capture(event):
        nonlocal is_capturing
        is_capturing = True
        print("Started capturing")

    def stop_capture(event):
        nonlocal is_capturing
        is_capturing = False
        print("Stopped capturing")

    # Create buttons
    cv2.createButton("Start", start_capture, None, cv2.QT_PUSH_BUTTON, 0)
    cv2.createButton("Stop", stop_capture, None, cv2.QT_PUSH_BUTTON, 0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Add status indicator
        status = "Recording" if is_capturing else "Stopped"
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 0, 255) if is_capturing else (255, 255, 255), 2)

        cv2.imshow("DroidCam Feed", frame)

        # Save frame if capturing
        if is_capturing:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_name = f"captured_image_{timestamp}_{image_counter}.jpg"
            image_path = os.path.join(cam_folder, image_name)
            
            if cv2.imwrite(image_path, frame):
                print(f"Image saved as {image_path}")
                saved_images.append(image_path)
                image_counter += 1

        if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
            break

    cap.release()
    cv2.destroyAllWindows()
    return saved_images
@app.route('/camera')
def camera():
    saved_images = camera_Access()
    return f"Images saved: {saved_images}"

#------------------------------------------------------------------------------------#
#---------------------------------video ACCESS-------------------------------#
#------------------------------------------------------------------------------------#

@app.route('/video_record')
def video_Access():

    from datetime import datetime
    import time
# Replace with your DroidCam IP and port
    #part1='1'
    #part2='27'
    #part1 = request.form.get('part1')
    #part2 = request.form.get('part2')
    #url='http://192.168.'+part1+'.'+part2+':4747/video'

    url = "http://192.168.1.11:4747/video"

# Open the video stream
    cap = cv2.VideoCapture(url)

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        exit()

# Get the frame width and height
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create a VideoWriter object
    output_file = "recorded_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format
    out = cv2.VideoWriter(output_file, fourcc, 20.0, (frame_width, frame_height))

# Record for 5 seconds
    start_time = time.time()
    while (time.time() - start_time) < 5:  # Record for 5 seconds
    # Read a frame from the stream
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

    # Write the frame to the output video file
        out.write(frame)

    # Display the frame (optional)
        cv2.imshow("Recording...", frame)

    # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release everything when done
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Video saved as {output_file}")
    return render_template('user/camera_status.html')
#------------------------------------------------------------------------------------------------------------#
#-----------------------------------------------UPLOAD VIDEO MANUALLY-------------------------------------------------#
#------------------------------------------------------------------------------------------------------------#
# Set the upload folder
@app.route('/upload', methods=['POST'])
def upload_video():
    UPLOAD_FOLDER = r'C:\Users\Lenovo\Desktop\final_project-main\uploads'
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    # Check if the post request has the file part
    if 'video' not in request.files:
        return "No file part in the request", 400

    file = request.files['video']

    # If the user does not select a file, the browser submits an empty file without a filename
    if file.filename == '':
        return "No selected file", 400

    # Save the file to the upload folder
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        return f"File uploaded successfully: {file.filename}", 200
    


#------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------#
#-----------------------------------------------GET AttENDANCE VIA IMAGE-------------------------------------------------#
#------------------------------------------------------------------------------------------------------------#
r"""def generate_file(path_of_detected_objects,class_name):
    # Model constants
    IMG_HEIGHT, IMG_WIDTH = 128, 128
    BATCH_SIZE = 16

    # Define training directory to get class labels
    train_dir = r"C:\Users\Lenovo\Desktop\final_project-main\datasets\preprocessed_data\train"

    # Create training generator to get class indices
    train_datagen = ImageDataGenerator(rescale=1./255)
    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    # Get class labels from training generator
    class_labels = list(train_gen.class_indices.keys())
    print("Class labels:", class_labels)

    # Load the model
    model_path = os.path.join(app.config['MODEL_PATH'], f"{class_name}")
    try:
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully")
    except Exception as e:
        raise Exception(f"Error loading model: {str(e)}")

    # Test images directory
    test_images_dir = path_of_detected_objects

    # Check if test_images_dir exists
    if not os.path.exists(test_images_dir):
        raise FileNotFoundError(f"Directory {test_images_dir} not found")

    # Image data generator for preprocessing
    test_datagen = ImageDataGenerator(rescale=1.0/255.0)

    # Load and preprocess test images
    def load_and_preprocess_images(image_dir):
        images = []
        filenames = []
        # Iterate over each image in the directory
        for image_name in os.listdir(image_dir):
            # Check if the file is an image
            if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    # Load and preprocess the image
                    image_path = os.path.join(image_dir, image_name)
            
                    image = load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))# Load image
                    image_array = img_to_array(image)# Convert to NumPy array
                    image_array = image_array / 255.0  # Normalize pixel values
                    #image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
                    # Append to the list
                    images.append(image_array)#
                    filenames.append(image_name)# Save the filename
                except Exception as e:
                    print(f"Error processing {image_name}: {str(e)}")
        
        return np.array(images), filenames

    # Initialize variables
    batch_images, batch_image_names = load_and_preprocess_images(test_images_dir)
    print(batch_images,batch_image_names)

    # Ensure there are images to process
    if not batch_images.size:
        raise ValueError("No valid images found in the directory.")

    # Predict
    predictions = model.predict(batch_images)
    predicted_classes = np.argmax(predictions, axis=1)
    confidence_scores = np.max(predictions, axis=1)  # Get confidence scores

    # Reverse the mapping of classes
    if 'train_gen' not in locals():
        raise ValueError("`train_gen` is not defined. Ensure `train_gen.class_indices` is available.")

    class_labels = {v: k for k, v in train_gen.class_indices.items()}#

    # Prepare data for Excel (with unique labels only)
    results = []
    unique_labels = set()
    i = 0
    # Generate unique roll numbers
    for image_name, predicted_class, confidence_score in zip(batch_image_names, predicted_classes, confidence_scores):# Iterate over each prediction
        predicted_label = class_labels.get(predicted_class, "Unknown")  # Handle unmapped classes
        if predicted_label not in unique_labels:  # Add only unique labels
            unique_labels.add(predicted_label)# Add to set
            confidence_percentage = round(confidence_score * 100, 2)  # Convert to percentage and round off
            i += 1
            results.append({
                "Predicted Label": predicted_label,
                "Accuracy (%)": confidence_percentage,
                "image_path": batch_image_names[i-1]
            })

    # Convert to a DataFrame
    results_df = pd.DataFrame(results)
    presents=list(unique_labels)
    actual=generate_members_list()
    emails=generate_emails(actual,domain = "adityatekkali.edu.in")
    print("Emails of Actual members:",emails)
    absents=[x for x in actual if x not in presents]
    emails=generate_emails(absents,domain = "adityatekkali.edu.in")
    print("Emails of absentees:",emails)
    #send_emails_to_absenties(emails)
    print("Emails sent to absentees")

    # Save to an Excel file
    excel_file_path = r"C:\Users\Lenovo\Desktop\final_project-main\result\Unique_Predicted_Results.xlsx"
    results_df.to_excel(excel_file_path, index=False)
    print(f"Unique results saved to {excel_file_path}")


@app.route('/get_attendance_video')
def get_attendance_image():
    extract_frames(r'C:\Users\Lenovo\Desktop\final_project-main\uploads\IMG_2800.MOV')
    #extract_frames(r'C:\Users\Lenovo\Desktop\final_project-main\recorded_video.mp4')
    #for i in range(10):
     #   detect_objects(r'C:\Users\Lenovo\Desktop\final_project-main\datap\frame_'+str(i)+'.jpg')
    detect_objects(r'C:\Users\Lenovo\Desktop\final_project-main\datap\frame_0.jpg')

    if request.args.get('class_name'):
        class_name = request.args.get('class_name')
        generate_file(r'C:\Users\Lenovo\Desktop\final_project-main\Detected_Objects', class_name)
        return display_excel()
    else:
        # Get list of available model files
        models_path = r'C:\Users\Lenovo\Desktop\final_project-main\models'
        model_files = [f.replace('.h5','') for f in os.listdir(models_path) if f.endswith('.h5')]
        # Show class selection form
        return render_template('user/select_class.html', model_files=model_files)
    return display_excel()
"""
#------------------------------------------------------------------------------------------------------------#

app.config['STATIC_FOLDER'] = 'static'  # Define static folder for serving images

@app.route('/display_excel')
def display_excel():
    excel_file_path = r"C:\Users\Lenovo\Desktop\final_project-main\result\Unique_Predicted_Results.xlsx"
    
    if not os.path.exists(excel_file_path):
        return "Error: Excel file not found. Please check the file path."
    
    # Read the Excel file
    df = pd.read_excel(excel_file_path)
    
    # Assuming images are moved to static/detected_objects folder
    static_image_dir = os.path.join(app.static_folder, 'detected_objects')
    if not os.path.exists(static_image_dir):
        os.makedirs(static_image_dir)
    
    # Copy images to static folder if not already there (you could automate this in generate_file)
    detected_objects_dir = r"C:\Users\Lenovo\Desktop\final_project-main\Detected_Objects"
    for img_file in os.listdir(detected_objects_dir):
        if img_file.endswith(('.jpg', '.png', '.jpeg')):
            src_path = os.path.join(detected_objects_dir, img_file)
            dest_path = os.path.join(static_image_dir, img_file)
            if not os.path.exists(dest_path):
                import shutil
                shutil.copy(src_path, dest_path)

    # Replace image_path with <img> tags pointing to static folder
    df['image_path'] = df['image_path'].apply(
        lambda x: f'<img src="/static/detected_objects/{x}" alt="Image" style="max-width: 100px; height: auto;">'
    )
    
    # Convert DataFrame to HTML with Bootstrap classes
    table_html = df.to_html(classes='table table-striped', index=False, escape=False)
    
    return render_template('user/display_excel.html', table=table_html)

def generate_file(path_of_detected_objects, class_name):
    # Model constants
    IMG_HEIGHT, IMG_WIDTH = 128, 128
    BATCH_SIZE = 16

    # Define training directory to get class labels
    train_dir = r"C:\Users\Lenovo\Desktop\final_project-main\datasets\preprocessed_data\train"

    # Create training generator to get class indices
    train_datagen = ImageDataGenerator(rescale=1./255)
    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    # Get class labels
    class_labels = list(train_gen.class_indices.keys())
    print("Class labels:", class_labels)

    # Load the model
    model_path = os.path.join(r"C:\Users\Lenovo\Desktop\final_project-main\models", f"{class_name}")
    try:
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully")
    except Exception as e:
        raise Exception(f"Error loading model: {str(e)}")

    # Test images directory
    test_images_dir = path_of_detected_objects
    if not os.path.exists(test_images_dir):
        raise FileNotFoundError(f"Directory {test_images_dir} not found")

    # Load and preprocess test images
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

    # Predict
    predictions = model.predict(batch_images)
    predicted_classes = np.argmax(predictions, axis=1)
    confidence_scores = np.max(predictions, axis=1)

    class_labels = {v: k for k, v in train_gen.class_indices.items()}

    # Prepare data for Excel
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
                "image_path": image_name  # Store just the filename
            })
    
    presents=list(unique_labels)
    actual=generate_members_list()
    emails=generate_emails(actual,domain = "adityatekkali.edu.in")
    print("Emails of Actual members:",emails)
    absents=[x for x in actual if x not in presents]
    emails=generate_emails(absents,domain = "adityatekkali.edu.in")
    print("Emails of absentees:",emails)
    #send_emails_to_absenties(emails)
    print("Emails sent to absentees")

    # Convert to DataFrame and save
    results_df = pd.DataFrame(results)
    excel_file_path = r"C:\Users\Lenovo\Desktop\final_project-main\result\Unique_Predicted_Results.xlsx"
    results_df.to_excel(excel_file_path, index=False)
    print(f"Unique results saved to {excel_file_path}")

@app.route('/get_attendance_video')
def get_attendance_image():
    # Placeholder functions (extract_frames, detect_objects) assumed to work
    extract_frames(r'C:\Users\Lenovo\Desktop\final_project-main\uploads\IMG_2800.MOV')
    detect_objects(r'C:\Users\Lenovo\Desktop\final_project-main\datap\frame_0.jpg')

    if request.args.get('class_name'):
        class_name = request.args.get('class_name')
        generate_file(r'C:\Users\Lenovo\Desktop\final_project-main\Detected_Objects', class_name)
        return display_excel()
    else:
        models_path = r"C:\Users\Lenovo\Desktop\final_project-main\models"
        model_files = [f.replace('.h5', '') for f in os.listdir(models_path) if f.endswith('.h5')]
        return render_template('user/select_class.html', model_files=model_files)

#------------------------------------------------------------------------------------------------------------#
#-----------------------------------------------DISPLAY EXCEL-------------------------------------------------#
#------------------------------------------------------------------------------------------------------------#
r"""@app.route('/display_excel')
def display_excel():
    excel_file_path = r"C:\Users\Lenovo\Desktop\final_project-main\result\Unique_Predicted_Results.xlsx"
    
    if not os.path.exists(excel_file_path):
        return "Error: Excel file not found. Please check the file path."
    
    df = pd.read_excel(excel_file_path)
    table_html = df.to_html(classes='table table-striped', index=False, escape=False)
    
    return render_template('user/display_excel.html', table=table_html)
"""

@app.route('/download_excel')
def download_excel():
    excel_file_path = r"C:\Users\Lenovo\Desktop\final_project-main\result\Unique_Predicted_Results.xlsx"
    
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
        models_path = r"C:\Users\Lenovo\Desktop\final_project-main\models"
        model_files = [f for f in os.listdir(models_path) if f.endswith('.h5')]
        model_files = [os.path.splitext(f)[0] for f in model_files]  # Remove file extension
        return render_template('user/models_list.html', models=model_files)
    except Exception as e:
        return f"Error accessing models: {str(e)}"
       
#------------------------------------------------------------------------------------------------------------#
#-----------------------------------------------RUN FLASK-------------------------------------------------#
#------------------------------------------------------------------------------------------------------------#

if __name__ == '__main__':
    app.run(debug=True)