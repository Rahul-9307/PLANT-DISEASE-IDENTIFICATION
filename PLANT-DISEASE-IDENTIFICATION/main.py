import streamlit as st
import tensorflow as tf
import numpy as np
import os
from PIL import Image
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase import pdfmetrics
import io
from datetime import datetime

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="AgriSens - Smart Plant Disease Detection",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
[data-testid="stSidebar"] {display: none;}
</style>
""", unsafe_allow_html=True)

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
# 38 CLASSES
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
# MULTI LANGUAGE INFO
# -------------------------------------------------
def get_info(language):
    if language == "Marathi":
        return {
            "description": "वनस्पतीमध्ये रोग आढळला आहे.",
            "cause": "हा रोग सहसा बुरशी किंवा जीवाणूंमुळे होतो.",
            "prevention": "योग्य पाणी व्यवस्थापन आणि स्वच्छता ठेवा.",
            "treatment": "कृषी तज्ञांचा सल्ला घ्या आणि योग्य फवारणी करा."
        }
    elif language == "Hindi":
        return {
            "description": "पौधे में रोग पाया गया है।",
            "cause": "यह रोग आमतौर पर फंगल या बैक्टीरियल संक्रमण से होता है।",
            "prevention": "सही सिंचाई और खेत की सफाई बनाए रखें।",
            "treatment": "कृषि विशेषज्ञ से सलाह लें और उचित दवा का छिड़काव करें।"
        }
    else:
        return {
            "description": "Disease detected in plant.",
            "cause": "Usually caused by fungal or bacterial infection.",
            "prevention": "Maintain proper irrigation and hygiene.",
            "treatment": "Consult agriculture expert and apply recommended fungicide."
        }

# -------------------------------------------------
# PREDICTION
# -------------------------------------------------
def model_prediction(test_image):
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    predictions = model.predict(input_arr)
    return np.argmax(predictions), np.max(predictions)*100

# -------------------------------------------------
# PDF GENERATOR
# -------------------------------------------------
def generate_pdf(disease, confidence, info, uploaded_image):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer)
    elements = []

    styles = getSampleStyleSheet()
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))

    normal = ParagraphStyle(
        name='Normal',
        parent=styles['Normal'],
        fontName="STSong-Light",
        fontSize=12
    )

    header = [[Paragraph("🌾 AGRISENS - PLANT DISEASE REPORT", normal)]]
    table = Table(header, colWidths=[450])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.green),
        ('TEXTCOLOR', (0,0), (-1,-1), colors.white),
        ('ALIGN',(0,0),(-1,-1),'CENTER')
    ]))
    elements.append(table)
    elements.append(Spacer(1, 0.3 * inch))

    elements.append(Paragraph(f"Date: {datetime.now().strftime('%d-%m-%Y %H:%M')}", normal))
    elements.append(Spacer(1, 0.3 * inch))

    if uploaded_image:
        img = RLImage(uploaded_image, width=4*inch, height=4*inch)
        img.hAlign = 'CENTER'
        elements.append(img)
        elements.append(Spacer(1, 0.3 * inch))

    summary = [
        ["Disease", disease],
        ["Confidence", f"{confidence:.2f}%"]
    ]

    summary_table = Table(summary, colWidths=[150,300])
    summary_table.setStyle(TableStyle([
        ('GRID',(0,0),(-1,-1),1,colors.grey)
    ]))

    elements.append(summary_table)
    elements.append(Spacer(1, 0.3 * inch))

    doc.build(elements)
    buffer.seek(0)
    return buffer

# -------------------------------------------------
# HERO SECTION
# -------------------------------------------------
if os.path.exists(IMAGE_PATH):
    st.markdown("""
    <div style="
        background: linear-gradient(135deg,#e8f5e9,#f1f8e9);
        padding:25px;
        border-radius:20px;
        box-shadow:0 8px 20px rgba(0,0,0,0.1);
        text-align:center;
        margin-bottom:30px;
    ">
    """, unsafe_allow_html=True)

    st.image(Image.open(IMAGE_PATH), use_column_width=True)

    st.markdown("""
        <h1 style="color:#2e7d32;">🌾 AgriSens - Smart Plant Disease Detection</h1>
        <p style="font-size:18px; color:#555;">
        Take Photo ➜ Upload ➜ Get Instant AI Diagnosis
        </p>
    </div>
    """, unsafe_allow_html=True)

# -------------------------------------------------
# MAIN SECTION
# -------------------------------------------------
language = st.selectbox(
    "Select Language / भाषा निवडा / भाषा चुनें",
    ["English", "Marathi", "Hindi"]
)

test_image = st.file_uploader("Upload Plant Leaf Image")

if test_image:
    st.image(test_image, use_column_width=True)

    if st.button("Predict"):

        index, confidence = model_prediction(test_image)
        disease = class_name[index].replace("___", " ")

        st.success(f"🌿 Prediction: {disease}")

        if confidence > 85:
            severity_text = "🔴 HIGH SEVERITY INFECTION"
            severity_color = "red"
        elif confidence > 60:
            severity_text = "🟠 MODERATE INFECTION LEVEL"
            severity_color = "orange"
        else:
            severity_text = "🟢 LOW INFECTION LEVEL"
            severity_color = "green"

        st.markdown(f"""
        <div style='background-color:{severity_color};
        padding:15px;border-radius:10px;
        text-align:center;color:white;
        font-size:22px;font-weight:bold;'>
        {severity_text}
        </div>
        """, unsafe_allow_html=True)

        st.info(f"📊 Confidence: {confidence:.2f}%")

        info = get_info(language)

        st.write("### Description")
        st.write(info["description"])
        st.write("### Cause")
        st.write(info["cause"])
        st.write("### Prevention")
        st.write(info["prevention"])
        st.write("### Treatment")
        st.write(info["treatment"])

        pdf = generate_pdf(disease, confidence, info, test_image)

        st.download_button(
            "📄 Download Report as PDF",
            data=pdf,
            file_name="AgriSens_Report.pdf",
            mime="application/pdf"
        )
