import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os

# Set page config
st.set_page_config(
    page_title="Space Station Safety Detector",
    page_icon="🚀",
    layout="wide"
)

# Title
st.title("🚀 Space Station Safety Object Detection")
st.markdown("---")

# Description
st.markdown("""
### Welcome to the Space Station Safety Equipment Detector!

This application uses YOLOv8 to detect critical safety equipment in space station environments.

**Detectable Objects:**
- 🔵 OxygenTank
- 🔵 NitrogenTank  
- 🟢 FirstAidBox
- 🔴 FireAlarm
- ⚡ SafetySwitchPanel
- ☎️ EmergencyPhone
- 🔥 FireExtinguisher
""")

# Load model
@st.cache_resource
def load_model():
    try:
        # Try to load custom model first
        if os.path.exists("best.pt"):
            return YOLO("best.pt")
        elif os.path.exists("yolov8s.pt"):
            return YOLO("yolov8s.pt")
        else:
            # Download and use YOLOv8 nano model
            return YOLO("yolov8n.pt")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load the model
model = load_model()

if model:
    st.success("✅ Model loaded successfully!")
    
    # File uploader
    st.header("📁 Upload Image for Detection")
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png"],
        help="Upload an image to detect safety objects"
    )
    
    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image, caption="Uploaded Image", use_container_width=True)
        
        with col2:
            st.subheader("Detection Results")
            
            # Run prediction
            with st.spinner("Analyzing image..."):
                results = model.predict(image, conf=0.5)
                
                # Get annotated image
                annotated_image = results[0].plot()
                # Convert BGR to RGB
                annotated_image = annotated_image[..., ::-1]
                
                st.image(annotated_image, caption="Detected Objects", use_container_width=True)
        
        # Show detection details
        st.subheader("📋 Detection Summary")
        
        detections = {}
        for box in results[0].boxes:
            class_name = model.names[int(box.cls[0])]
            confidence = float(box.conf[0])
            detections[class_name] = detections.get(class_name, 0) + 1
        
        if detections:
            st.write("**Detected Objects:**")
            for obj, count in detections.items():
                st.write(f"- **{obj}**: {count} detected")
        else:
            st.info("No objects detected in this image.")
            
else:
    st.error("❌ Failed to load model. Please check the model files.")

# Footer
st.markdown("---")
st.markdown("""
**About this Application:**
- Built with YOLOv8 and Streamlit
- Designed for space station safety monitoring
- Part of the Duality AI Hackathon project
""")