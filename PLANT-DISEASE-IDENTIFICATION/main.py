import streamlit as st
import tensorflow as tf
import numpy as np
import os
from PIL import Image
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase import pdfmetrics
import io

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="AgriSens - Smart Plant Disease Detection",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Hide Sidebar
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
# LOAD MODEL
# -------------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# -------------------------------------------------
# FULL 38 CLASSES
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
# GENERIC ACTION PLAN (MULTI LANGUAGE)
# -------------------------------------------------
def get_info(language):

    if language == "English":
        return {
            "description": "Disease detected in plant.",
            "cause": "Usually caused by fungal or bacterial infection.",
            "prevention": "Maintain proper irrigation and field hygiene.",
            "treatment": "Consult agriculture expert and apply recommended fungicide."
        }

    elif language == "Marathi":
        return {
            "description": "वनस्पतीमध्ये रोग आढळला आहे.",
            "cause": "हा रोग सहसा बुरशी किंवा जीवाणूंमुळे होतो.",
            "prevention": "योग्य पाणी व्यवस्थापन आणि स्वच्छता ठेवा.",
            "treatment": "कृषी तज्ञांचा सल्ला घ्या आणि योग्य फवारणी करा."
        }

    else:
        return {
            "description": "पौधे में रोग पाया गया है।",
            "cause": "यह रोग आमतौर पर फंगल या बैक्टीरियल संक्रमण से होता है।",
            "prevention": "सही सिंचाई और खेत की सफाई बनाए रखें।",
            "treatment": "कृषि विशेषज्ञ से सलाह लें और उचित दवा का छिड़काव करें।"
        }

# -------------------------------------------------
# PREDICTION FUNCTION
# -------------------------------------------------
def model_prediction(test_image):
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    predictions = model.predict(input_arr)
    return np.argmax(predictions), np.max(predictions)*100

# -------------------------------------------------
# ATTRACTIVE UNICODE PDF (CLOUD SAFE)
# -------------------------------------------------
def generate_pdf(disease, confidence, info):

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer)
    elements = []

    styles = getSampleStyleSheet()

    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))

    normal_style = ParagraphStyle(
        name='NormalStyle',
        parent=styles['Normal'],
        fontName="STSong-Light",
        fontSize=12,
        spaceAfter=10
    )

    title_style = ParagraphStyle(
        name='TitleStyle',
        parent=styles['Heading1'],
        fontName="STSong-Light",
        fontSize=18,
        textColor=colors.green,
        spaceAfter=20
    )

    elements.append(Paragraph("AgriSens - Plant Disease Report", title_style))
    elements.append(Spacer(1, 0.3 * inch))

    elements.append(Paragraph(f"<b>Disease:</b> {disease}", normal_style))
    elements.append(Paragraph(f"<b>Confidence:</b> {confidence:.2f}%", normal_style))
    elements.append(Spacer(1, 0.2 * inch))

    elements.append(Paragraph(f"<b>Description:</b> {info['description']}", normal_style))
    elements.append(Paragraph(f"<b>Cause:</b> {info['cause']}", normal_style))
    elements.append(Paragraph(f"<b>Prevention:</b> {info['prevention']}", normal_style))
    elements.append(Paragraph(f"<b>Treatment:</b> {info['treatment']}", normal_style))

    doc.build(elements)
    buffer.seek(0)
    return buffer

# -------------------------------------------------
# UI
# -------------------------------------------------
col1, col2, col3 = st.columns([1,3,1])

with col2:

    if os.path.exists(IMAGE_PATH):
        st.image(Image.open(IMAGE_PATH), use_column_width=True)

    language = st.selectbox(
        "Select Language / भाषा निवडा / भाषा चुनें",
        ["English", "Marathi", "Hindi"]
    )

    test_image = st.file_uploader("Upload Plant Leaf Image")

    if test_image:
        st.image(test_image, use_column_width=True)

        if st.button("Predict"):

            index, confidence = model_prediction(test_image)

            if index >= len(class_name):
                st.error("Prediction error.")
                st.stop()

            disease_name = class_name[index].replace("___", " ")

            st.success(f"Prediction: {disease_name}")
            st.info(f"Confidence: {confidence:.2f}%")

            # Severity
            if confidence > 85:
                st.error("High Severity Infection!")
            elif confidence > 60:
                st.warning("Moderate Infection Level")
            else:
                st.success("Low Infection Level")

            info = get_info(language)

            st.write("### Description")
            st.write(info["description"])
            st.write("### Cause")
            st.write(info["cause"])
            st.write("### Prevention")
            st.write(info["prevention"])
            st.write("### Treatment")
            st.write(info["treatment"])

            pdf = generate_pdf(disease_name, confidence, info)

            st.download_button(
                "Download Report as PDF",
                data=pdf,
                file_name="plant_disease_report.pdf",
                mime="application/pdf"
            )
