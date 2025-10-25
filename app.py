import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os
import pandas as pd
from pathlib import Path
import time
import cv2
import requests
import gdown
from tqdm import tqdm

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Workplace Safety Detector",
    page_icon="⚠️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR A BEAUTIFUL UI ---
st.markdown("""
<style>
    /* Main app background */
    .stApp {
        background-color: #ffffff;
    }
    /* Custom title styling */
    .title-text {
        color: #1E3A8A; /* Deep blue */
        text-align: center;
        font-size: 2.5rem;
        font-weight: bold;
    }
    /* Ensure all text is visible */
    .stMarkdown, .stText, p, div, span {
        color: #000000 !important;
        text-align: justify;
    }
    /* Custom button styling */
    .stButton>button {
        background-color: #2563EB; /* Bright blue */
        color: white !important;
        border-radius: 12px;
        padding: 10px 20px;
        border: none;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    /* Sidebar styling */
    .css-1d391kg, .css-1544g2n {
        background-color: #f8f9fa;
        color: #000000 !important;
    }
    /* File uploader styling */
    .stFileUploader > div {
        color: #000000 !important;
    }
</style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---
@st.cache_resource
def load_model(model_path):
    """Loads the YOLOv8 model from the specified path."""
    # Model configuration
    MODEL_FILENAME = "best.pt"

    # Handle model file with spaces in the name
    if os.path.exists("best (1).pt") and not os.path.exists(MODEL_FILENAME):
        # Rename the file to avoid spaces
        os.rename("best (1).pt", MODEL_FILENAME)

    # Load the YOLO model
    model = None
    if os.path.exists(MODEL_FILENAME):
        try:
            model = YOLO(MODEL_FILENAME)
            st.session_state.model_loaded = True
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            st.session_state.model_loaded = False
    else:
        st.error(f"Model file '{MODEL_FILENAME}' not found. Please upload the model file.")
        st.session_state.model_loaded = False

    return model

def find_latest_run_dir(base_dir="runs/detect"):
    """Finds the latest training run directory."""
    if not os.path.exists(base_dir):
        return None
    train_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith('train')]
    if not train_dirs:
        return None
    return os.path.join(base_dir, max(train_dirs)) # Returns the folder with the highest number, e.g., 'train3'

# --- MAIN APPLICATION ---

# --- HEADER ---
st.markdown('<p class="title-text">⚠️ Workplace Safety Equipment Detector</p>', unsafe_allow_html=True)
st.markdown("---")
st.markdown("""
Welcome to an advanced object detection application powered by **YOLOv8**. This tool is designed to identify critical safety equipment in any workspace. 
Simply upload an image, and the AI will analyze it to detect and classify objects based on its training.
""")

# --- MODEL AND RESULTS PATHS ---
# Use the best (1).pt file in the current directory
MODEL_PATH = "best (1).pt"
LATEST_RUN_DIR = None  # We'll skip the training results section

# --- SIDEBAR ---
with st.sidebar:
    st.header("📖 About")
    
    # Define the actual classes from your model
    model_classes = [
        "OxygenTank",
        "NitrogenTank",
        "FirstAidBox",
        "FireAlarm",
        "SafetySwitchPanel",
        "EmergencyPhone",
        "FireExtinguisher"
    ]
    
    # Create the info text
    about_text = """
    This application uses a custom YOLOv8 model to detect safety equipment in images.
    """
    
    st.info(about_text)

# --- MODEL LOADING ---
model = load_model(MODEL_PATH)

# --- IMAGE SOURCE SELECTION ---
st.header("📷 Image Source")

# Create tabs for different input methods
tab1, tab2 = st.tabs(["📁 Upload Images", "📸 Use Camera"])

with tab1:
    st.subheader("Upload Images for Detection")
    uploaded_files = st.file_uploader(
        "Drag and drop one or more image files, or click to browse.",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="file_uploader"
    )
    
    # Process uploaded images
    if uploaded_files and model:
        # Create a progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Process each uploaded file
        for i, uploaded_file in enumerate(uploaded_files):
            try:
                # Update progress
                progress = (i + 1) / len(uploaded_files)
                progress_bar.progress(progress)
                status_text.text(f"Processing image {i + 1} of {len(uploaded_files)}...")
                
                # Open and process the image
                image = Image.open(uploaded_file)
                
                # Create columns for original and result
                col1, col2 = st.columns(2)
                
                with col1:
                    st.image(image, caption=f"Original: {uploaded_file.name}", use_container_width=True)
                
                with col2:
                    with st.spinner(f"Analyzing {uploaded_file.name}..."):
                        # Run prediction
                        results = model.predict(image, conf=0.5)
                        
                        # Create annotated image
                        annotated_image = results[0].plot()[..., ::-1]  # Convert BGR to RGB
                        st.image(annotated_image, caption=f"Results: {uploaded_file.name}", use_container_width=True)
                        
                        # Show detection summary
                        with st.expander(f"📋 Detection Details: {uploaded_file.name}"):
                            detections = {}
                            for box in results[0].boxes:
                                class_name = model.names[int(box.cls[0])]
                                detections[class_name] = detections.get(class_name, 0) + 1
                            
                            if detections:
                                st.write("**Detected Objects:**")
                                for item, count in detections.items():
                                    st.write(f"- {item}: {count}")
                            else:
                                st.info("No objects detected in this image.")
                            
                            st.markdown("---")
                
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
        
        # Update progress to complete
        if uploaded_files:
            progress_bar.progress(1.0)
            status_text.success("✅ All images processed!")
    elif uploaded_files and not model:
        st.error("Model failed to load. Please check the model file.")

with tab2:
    # Real-time camera feed with detection
    st.subheader("Real-time Camera Detection")
    
    # Add a checkbox to start/stop the camera
    run_camera = st.checkbox('Start Camera', key='run_camera')
    
    if run_camera:
        # Use OpenCV for camera capture
        import cv2
        
        # Initialize the camera
        cap = cv2.VideoCapture(0)
        
        # Create placeholders
        camera_placeholder = st.empty()
        result_placeholder = st.empty()
        
        # Create a stop button
        stop_button = st.button('Stop Camera')
        
        while run_camera and not stop_button and cap.isOpened():
            # Read frame from camera
            ret, frame = cap.read()
            
            if ret:
                # Convert the image from BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Create a PIL Image from the frame
                image = Image.fromarray(frame_rgb)
                
                try:
                    # Run prediction
                    results = model.predict(image, conf=0.5)
                    
                    # Convert frame to numpy array for drawing
                    frame_np = frame_rgb.copy()
                    
                    # Draw detection boxes directly on the frame
                    for box in results[0].boxes:
                        # Get box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        # Get class name and confidence
                        class_id = int(box.cls[0])
                        class_name = model.names[class_id]
                        confidence = float(box.conf[0])
                        
                        # Draw rectangle
                        color = (0, 255, 0)  # Green color for boxes
                        cv2.rectangle(frame_np, (x1, y1), (x2, y2), color, 2)
                        
                        # Create label with class name and confidence
                        label = f"{class_name} {confidence:.2f}"
                        
                        # Get text size
                        (text_width, text_height), _ = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        
                        # Draw filled rectangle for label background
                        cv2.rectangle(
                            frame_np, 
                            (x1, y1 - text_height - 5), 
                            (x1 + text_width, y1), 
                            color, 
                            -1
                        )
                        
                        # Put text on the image
                        cv2.putText(
                            frame_np,
                            label,
                            (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 0, 0),  # Black text
                            1,
                            cv2.LINE_AA
                        )
                    
                    # Convert back to PIL Image for display
                    annotated_image = Image.fromarray(frame_np)
                    
                    # Display the annotated frame
                    camera_placeholder.image(annotated_image, caption='Live Detection', use_column_width=True)
                    
                    # Show detection summary
                    detections = {}
                    for box in results[0].boxes:
                        class_name = model.names[int(box.cls[0])]
                        detections[class_name] = detections.get(class_name, 0) + 1
                    
                    # Update the detection summary
                    if detections:
                        detection_text = "**Detected Objects:**\n"
                        for item, count in detections.items():
                            detection_text += f"- {item}: {count}\n"
                        result_placeholder.markdown(detection_text)
                    else:
                        result_placeholder.info("No objects detected")
                    
                except Exception as e:
                    result_placeholder.error(f"Error during detection: {str(e)}")
                    camera_placeholder.image(image, caption='Live Camera Feed', use_column_width=True)
                
                # Add a small delay to control the frame rate
                time.sleep(0.05)  # ~20 FPS
                
            # Check if stop button was clicked or camera failed
            if stop_button or not ret:
                break
        
        # Release the camera when done
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        
        # Clear the placeholders when stopped
        if stop_button:
            camera_placeholder.empty()
            result_placeholder.empty()
            st.success("Camera stopped")
    else:
        st.info("Check the 'Start Camera' checkbox to begin real-time detection")

# Removed duplicate code block that was causing issues

# --- MODEL DOWNLOAD FUNCTION ---
@st.cache_resource
def download_model():
    """Download the model file from Google Drive if it doesn't exist"""
    # Google Drive direct download link (generated from the shareable link)
    MODEL_URL = "https://drive.google.com/uc?export=download&id=1Ske7AZgMLtyWvv076iqtlCBcc9Kwmjfe"
    MODEL_FILENAME = "best.pt"
    
    # Check if model already exists
    if os.path.exists(MODEL_FILENAME):
        return MODEL_FILENAME
    
    # Show download progress
    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()
    status_text.text("Downloading model... (This may take a few minutes)")
    
    try:
        # Create a session to handle cookies
        session = requests.Session()
        
        # First request to get the confirmation token
        response = session.get(MODEL_URL, stream=True)
        token = ""
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                token = value
                break
        
        # Second request with the token
        if token:
            params = {'id': '1Ske7AZgMLtyWvv076iqtlCBcc9Kwmjfe', 'confirm': token}
            response = session.get('https://drive.google.com/uc', params=params, stream=True)
        
        # Get the file size for progress tracking
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 * 1024  # 1MB
        downloaded_size = 0
        
        # Download the file with progress
        with open(MODEL_FILENAME, 'wb') as f:
            for data in response.iter_content(block_size):
                downloaded_size += len(data)
                f.write(data)
                # Update progress
                progress = min(downloaded_size / total_size, 1.0)
                progress_bar.progress(progress)
        
        status_text.success("Model downloaded successfully!")
        progress_bar.empty()
        return MODEL_FILENAME
        
    except Exception as e:
        status_text.error(f"Error downloading model: {str(e)}")
        progress_bar.empty()
        return None

# --- MODEL LOADING ---
st.sidebar.header("Model Configuration")

# Download and load the model
MODEL_FILENAME = download_model()

# Load the YOLO model
model = None
if MODEL_FILENAME and os.path.exists(MODEL_FILENAME):
    try:
        with st.spinner('Loading model...'):
            model = YOLO(MODEL_FILENAME)
            st.session_state.model_loaded = True
            st.sidebar.success("Model loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Error loading model: {str(e)}")
        st.session_state.model_loaded = False
else:
    st.sidebar.error("Failed to download or find the model file.")
    st.session_state.model_loaded = False

# --- TRAINING RESULTS SECTION ---
st.markdown("---")
st.header("📊 Model Information")

# Display model evaluation metrics and graphs
if os.path.exists("images"):
    st.subheader("Model Performance Metrics")
    
    # Create columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        if os.path.exists("images/results.png"):
            st.image("images/results.png", caption="Training Results", use_container_width=True)
        if os.path.exists("images/BoxF1_curve.png"):
            st.image("images/BoxF1_curve.png", caption="F1 Score Curve", use_container_width=True)
    
    with col2:
        if os.path.exists("images/confusion_matrix.png"):
            st.image("images/confusion_matrix.png", caption="Confusion Matrix", use_container_width=True)
        if os.path.exists("images/confusion_matrix_normalized.png"):
            st.image("images/confusion_matrix_normalized.png", caption="Normalized Confusion Matrix", use_container_width=True)
else:
    st.warning("No evaluation images found. Please ensure the 'images' folder exists with model evaluation images.")

# Display model information
st.info("""
This application is using a custom YOLOv8 model for object detection.
The model is specifically trained to detect safety equipment in workplace environments.

**Model Evaluation Metrics:**
- The F1 score curve shows the balance between precision and recall
- Confusion matrices display the model's performance across different classes
- Training results include metrics like mAP (mean Average Precision)
""")

