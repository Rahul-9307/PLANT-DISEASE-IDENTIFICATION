import streamlit as st
import tensorflow as tf
import numpy as np
import os
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io

# -------------------------------------------------
# PAGE CONFIG (NO SIDEBAR)
# -------------------------------------------------
st.set_page_config(
    page_title="AgriSens - Disease Detection",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Hide sidebar completely
st.markdown("""
<style>
[data-testid="stSidebar"] {display: none;}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# PATHS
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_PATH = os.path.join(BASE_DIR, "Diseases.png")
MODEL_PATH = os.path.join(BASE_DIR, "trained_plant_disease_model.keras")

# -------------------------------------------------
# LOAD MODEL (Cached)
# -------------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# -------------------------------------------------
# CLASS NAMES
# -------------------------------------------------
class_name = [
    'Apple___Apple_scab','Apple___Black_rot','Apple___Cedar_apple_rust','Apple___healthy',
    'Blueberry___healthy','Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy','Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_','Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy','Grape___Black_rot',
    'Grape___Esca_(Black_Measles)','Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy','Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot','Peach___healthy',
    'Pepper,_bell___Bacterial_spot','Pepper,_bell___healthy',
    'Potato___Early_blight','Potato___Late_blight','Potato___healthy',
    'Raspberry___healthy','Soybean___healthy','Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch','Strawberry___healthy',
    'Tomato___Bacterial_spot','Tomato___Early_blight',
    'Tomato___Late_blight','Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# -------------------------------------------------
# DISEASE INFO (Sample – You Can Expand)
# -------------------------------------------------
disease_info = {
    "Tomato___Early_blight": {
        "description": "Fungal disease causing brown spots on leaves.",
        "prevention": "Use fungicide and avoid overhead watering."
    },
    "Tomato___healthy": {
        "description": "The plant is healthy.",
        "prevention": "Maintain proper irrigation and fertilization."
    }
}

# -------------------------------------------------
# PREDICTION FUNCTION
# -------------------------------------------------
def model_prediction(test_image):
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    predictions = model.predict(input_arr)

    result_index = np.argmax(predictions)
    confidence = np.max(predictions) * 100

    return result_index, confidence

# -------------------------------------------------
# PDF GENERATION
# -------------------------------------------------
def generate_pdf(disease, confidence):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)

    c.drawString(100, 750, "AgriSens - Plant Disease Report")
    c.drawString(100, 720, f"Disease: {disease}")
    c.drawString(100, 700, f"Confidence: {confidence:.2f}%")

    if disease in disease_info:
        c.drawString(100, 670, "Description:")
        c.drawString(100, 650, disease_info[disease]["description"])

        c.drawString(100, 620, "Prevention:")
        c.drawString(100, 600, disease_info[disease]["prevention"])

    c.save()
    buffer.seek(0)
    return buffer

# -------------------------------------------------
# CENTER LAYOUT
# -------------------------------------------------
col1, col2, col3 = st.columns([1,3,1])

with col2:

    # Banner
    if os.path.exists(IMAGE_PATH):
        img = Image.open(IMAGE_PATH)
        st.image(img, use_column_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    app_mode = st.selectbox("Select a Page", ["HOME", "DISEASE RECOGNITION"])

    st.markdown("<br>", unsafe_allow_html=True)

    # HOME
    if app_mode == "HOME":

        st.markdown(
            "<h1 style='text-align: center; color: #2e8b57;'>SMART PLANT DISEASE DETECTION</h1>",
            unsafe_allow_html=True
        )

        st.markdown(
            "<p style='text-align: center;'>Upload a plant leaf image and our AI model will detect the disease with confidence score and prevention tips.</p>",
            unsafe_allow_html=True
        )

    # DISEASE PAGE
    elif app_mode == "DISEASE RECOGNITION":

        st.markdown(
            "<h1 style='text-align: center; color: #4CAF50;'>Disease Recognition</h1>",
            unsafe_allow_html=True
        )

        st.markdown("<br>", unsafe_allow_html=True)

        test_image = st.file_uploader("Choose an Image of the Plant:")

        if test_image is not None:
            st.image(test_image, use_column_width=True)

            if st.button("Predict"):

                result_index, confidence = model_prediction(test_image)
                disease_name = class_name[result_index]

                st.success(f"🌿 Prediction: {disease_name}")
                st.info(f"🔍 Confidence: {confidence:.2f}%")

                if disease_name in disease_info:
                    st.write("### 🦠 Description")
                    st.write(disease_info[disease_name]["description"])

                    st.write("### 🌱 Prevention")
                    st.write(disease_info[disease_name]["prevention"])

                # PDF Download
                pdf = generate_pdf(disease_name, confidence)

                st.download_button(
                    label="📄 Download Report as PDF",
                    data=pdf,
                    file_name="plant_disease_report.pdf",
                    mime="application/pdf"
                )
