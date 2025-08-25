import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import os
import base64
import folium
from datetime import datetime
from streamlit.components.v1 import html
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from fpdf import FPDF
import matplotlib.pyplot as plt

# === SET PAGE CONFIG FIRST - MUST BE FIRST STREAMLIT COMMAND ===
st.set_page_config(page_title="Plant Disease Classifier", page_icon="🌿", layout="wide")

# === Absolute Paths ===
BASE_DIR = os.getcwd()
MODEL_PATH = os.path.join(BASE_DIR, "Iss_outputs", "plant_disease_model.keras")
MAP_HTML = os.path.join(BASE_DIR, "Iss_outputs", "interactive_plant_map.html")
PDF_REPORT_PATH = os.path.join(BASE_DIR, "Iss_outputs", "plant_report.pdf")
HISTORY_CSV_PATH = os.path.join(BASE_DIR, "Iss_outputs", "prediction_history.csv")
IMG_SIZE = (224, 224)

# === Helper Functions (NO STREAMLIT COMMANDS HERE) ===
def preprocess_image(img):
    img = img.resize(IMG_SIZE)
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    return img_array

def encode_image_to_base64(img):
    from io import BytesIO
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def generate_map(lat, lon, status, img):
    m = folium.Map(location=[lat, lon], zoom_start=6)
    color = "red" if status == "Infected" else "green"
    
    img_resized = img.copy()
    img_resized.thumbnail((150, 150))
    img_b64 = encode_image_to_base64(img_resized)
    
    popup = f"<b>Status:</b> {status}<br>({lat},{lon})<br><img src='data:image/jpeg;base64,{img_b64}' width='150'>"
    folium.Marker([lat, lon], popup=popup, icon=folium.Icon(color=color)).add_to(m)
    m.save(MAP_HTML)

def generate_pdf(status, lat, lon, confidence, filename):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=16, style='B')
    pdf.cell(0, 10, "Plant Disease Report", ln=True, align='C')
    pdf.ln(10)
    
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"Image: {filename}", ln=True)
    pdf.cell(0, 10, f"Status: {status}", ln=True)
    pdf.cell(0, 10, f"Confidence: {confidence:.2%}", ln=True)
    pdf.cell(0, 10, f"Latitude: {lat}", ln=True)
    pdf.cell(0, 10, f"Longitude: {lon}", ln=True)
    pdf.cell(0, 10, f"Date: {datetime.now()}", ln=True)
    
    pdf.output(PDF_REPORT_PATH)

def save_history(entry):
    os.makedirs(os.path.dirname(HISTORY_CSV_PATH), exist_ok=True)
    if os.path.exists(HISTORY_CSV_PATH):
        df = pd.read_csv(HISTORY_CSV_PATH)
        df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
    else:
        df = pd.DataFrame([entry])
    df.to_csv(HISTORY_CSV_PATH, index=False)

# === Load the trained model ===
@st.cache_resource
def load_cached_model():
    try:
        # Check if model file exists first
        if not os.path.exists(MODEL_PATH):
            st.error(f"❌ Model file not found at: {MODEL_PATH}")
            # Try alternative path with .h5 extension
            alternative_path = MODEL_PATH.replace('.keras', '.h5')
            if os.path.exists(alternative_path):
                st.info(f"Trying alternative path: {alternative_path}")
                model = load_model(alternative_path, compile=False)
            else:
                return None
        else:
            model = load_model(MODEL_PATH, compile=False)
        
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

# === MAIN APP CODE ===
st.title("🌿 Plant Disease Classifier")
st.markdown("Upload plant images to check if they're healthy or infected.")

# Load model - this happens after page config
model = load_cached_model()

if model is None:
    st.stop()

def classify_image(img_array):
    img_array = np.expand_dims(img_array, axis=0)
    probability = model.predict(img_array, verbose=0)[0][0]
    predicted_class = 1 if probability > 0.5 else 0
    confidence = probability if predicted_class == 1 else 1 - probability
    return predicted_class, confidence

def process_single_image(image_file, lat, lon):
    """Process a single uploaded image"""
    image = Image.open(image_file).convert('RGB')
    img_array = preprocess_image(image)
    label, confidence = classify_image(img_array)
    status = "Infected" if label == 1 else "Healthy"
    
    return {
        'image': image,
        'status': status,
        'confidence': confidence,
        'filename': image_file.name,
        'lat': lat,
        'lon': lon
    }

def process_batch_images(image_files, lat, lon):
    """Process multiple images in batch"""
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, image_file in enumerate(image_files):
        status_text.text(f"Processing {i+1}/{len(image_files)}: {image_file.name}")
        progress_bar.progress((i + 1) / len(image_files))
        
        try:
            result = process_single_image(image_file, lat, lon)
            results.append(result)
        except Exception as e:
            st.error(f"Error processing {image_file.name}: {str(e)}")
    
    progress_bar.empty()
    status_text.empty()
    return results

# === Upload Options ===
upload_option = st.radio("Choose upload method:", 
                         ["Single Image", "Multiple Images"])

if upload_option == "Single Image":
    uploaded_files = st.file_uploader("Upload a plant image", 
                                     type=["jpg", "jpeg", "png"],
                                     accept_multiple_files=False)
else:
    uploaded_files = st.file_uploader("Upload multiple plant images", 
                                     type=["jpg", "jpeg", "png"],
                                     accept_multiple_files=True)

col1, col2 = st.columns(2)
with col1:
    lat = st.number_input("Latitude", value=36.0, format="%.6f")
with col2:
    lon = st.number_input("Longitude", value=3.0, format="%.6f")

col1, col2 = st.columns(2)
with col1:
    classify_clicked = st.button("🔍 Classify", type="primary", use_container_width=True)
with col2:
    reset_clicked = st.button("🗑️ Reset History", use_container_width=True)

# === Reset CSV ===
if reset_clicked:
    if os.path.exists(HISTORY_CSV_PATH):
        os.remove(HISTORY_CSV_PATH)
        st.success("Prediction history cleared!")
    else:
        st.info("No history file found to delete.")

# === Classification Logic ===

if classify_clicked:
    if uploaded_files:
        if upload_option == "Single Image":
            results = [process_single_image(uploaded_files, lat, lon)]
        else:
            results = process_batch_images(uploaded_files, lat, lon)
        
        # Display results
        if results:
            st.subheader("📊 Classification Results")
            
            for i, result in enumerate(results):
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    # FIXED: Removed use_container_width parameter
                    st.image(result['image'], caption=result['filename'])
                
                with col2:
                    if result['status'] == "Healthy":
                        st.success(f"**Prediction:** {result['status']} ({result['confidence']:.2%} confidence)")
                    else:
                        st.error(f"**Prediction:** {result['status']} ({result['confidence']:.2%} confidence)")
                    
                    st.write(f"**Location:** ({result['lat']:.4f}, {result['lon']:.4f})")
                
                if i < len(results) - 1:
                    st.divider()
                
                # Save to history
                save_history({
                    "Datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Status": result['status'],
                    "Confidence": result['confidence'],
                    "Latitude": result['lat'],
                    "Longitude": result['lon'],
                    "Image_File": result['filename']
                })
            #******************************************************************
            # Generate map and PDF for first result
            generate_map(results[0]['lat'], results[0]['lon'], results[0]['status'], results[0]['image'])
            
            try:
                with open(MAP_HTML, "r", encoding="utf-8") as f:
                    map_html = f.read()
                html(map_html, height=500)
            except:
                st.warning("Could not display map")
            
            generate_pdf(results[0]['status'], results[0]['lat'], results[0]['lon'], 
                        results[0]['confidence'], results[0]['filename'])
            
            with open(PDF_REPORT_PATH, "rb") as f:
                st.download_button("📄 Download PDF Report", f, file_name="plant_report.pdf")
            
            # Batch statistics
            if len(results) > 1:
                st.subheader("📈 Batch Statistics")
                status_counts = pd.Series([r['status'] for r in results]).value_counts()
                
                fig, ax = plt.subplots()
                ax.pie(status_counts.values, labels=status_counts.index, autopct='%1.1f%%', 
                      colors=['green', 'red'], startangle=90)
                ax.axis('equal')
                st.pyplot(fig)
                
                st.write(f"**Total Images:** {len(results)}")
                st.write(f"**Healthy:** {status_counts.get('Healthy', 0)}")
                st.write(f"**Infected:** {status_counts.get('Infected', 0)}")
    else:
        st.warning("⚠️ Please upload an image first!")

# === Display History ===
if os.path.exists(HISTORY_CSV_PATH):
    st.subheader("📁 Classification History")
    df = pd.read_csv(HISTORY_CSV_PATH)
    st.dataframe(df)
    
    if not df.empty:
        st.subheader("📊 History Analytics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            healthy_count = len(df[df['Status'] == 'Healthy'])
            st.metric("Healthy Plants", healthy_count)
        with col3:
            infected_count = len(df[df['Status'] == 'Infected'])
            st.metric("Infected Plants", infected_count)
else:
    st.info("No prediction history yet. Classify some images to see data here!")

# === Footer ===
st.markdown("---")
st.caption("🌿 Plant Disease Classifier | Built with Streamlit & TensorFlow | Enjoy by Smile Bey")