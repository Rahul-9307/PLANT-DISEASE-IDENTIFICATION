import streamlit as st
import tensorflow as tf
import numpy as np
import os
from PIL import Image
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
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
    page_title="Agri🌾Next - Smart Plant Disease Detection",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
[data-testid="stSidebar"] {display: none;}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# PATHS
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "trained_plant_disease_model.keras")

# -------------------------------------------------
# LOAD MODEL
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
# MULTI LANGUAGE INFO
# -------------------------------------------------
def get_info(language):
    if language == "Marathi":
        return {
            "description": "वनस्पतीमध्ये रोग आढळला आहे.",
            "cause": "हा रोग बुरशी किंवा जीवाणूंमुळे होतो.",
            "prevention": "योग्य पाणी व्यवस्थापन आणि स्वच्छता ठेवा.",
            "treatment": "कृषी तज्ञांचा सल्ला घ्या."
        }
    elif language == "Hindi":
        return {
            "description": "पौधे में रोग पाया गया है।",
            "cause": "यह रोग फंगल या बैक्टीरियल संक्रमण से होता है।",
            "prevention": "सही सिंचाई और खेत की सफाई रखें।",
            "treatment": "कृषि विशेषज्ञ से सलाह लें।"
        }
    else:
        return {
            "description": "Disease detected in plant.",
            "cause": "Usually caused by fungal or bacterial infection.",
            "prevention": "Maintain proper irrigation and hygiene.",
            "treatment": "Consult agriculture expert."
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
        spaceAfter=8
    )

    header = [[Paragraph("🌾 AGRINEXT - PLANT DISEASE REPORT", normal_style)]]
    header_table = Table(header, colWidths=[450])
    header_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.green),
        ('TEXTCOLOR', (0,0), (-1,-1), colors.white),
        ('ALIGN',(0,0),(-1,-1),'CENTER'),
    ]))

    elements.append(header_table)
    elements.append(Spacer(1, 0.3 * inch))

    summary = [
        ["Date", datetime.now().strftime('%d-%m-%Y %H:%M')],
        ["Disease", disease],
        ["Confidence", f"{confidence:.2f}%"]
    ]

    table = Table(summary, colWidths=[150,300])
    table.setStyle(TableStyle([
        ('GRID',(0,0),(-1,-1),1,colors.grey),
        ('FONTNAME',(0,0),(-1,-1),"STSong-Light"),
    ]))

    elements.append(table)
    elements.append(Spacer(1, 0.3 * inch))

    for key, value in info.items():
        elements.append(Paragraph(f"<b>{key.capitalize()}:</b> {value}", normal_style))
        elements.append(Spacer(1, 0.2 * inch))

    doc.build(elements)
    buffer.seek(0)
    return buffer

# -------------------------------------------------
# UI CENTER
# -------------------------------------------------
col1, col2, col3 = st.columns([1,3,1])

with col2:

    # MULTIPLE BANNERS
    image_list = ["Diseases.png", "banner2.png", "banner3.png"]

    for img_name in image_list:
        img_path = os.path.join(BASE_DIR, img_name)
        if os.path.exists(img_path):
            st.markdown("""
            <div style="
                background-color:#0e1117;
                padding:15px;
                border-radius:15px;
                margin-bottom:20px;
                box-shadow:0px 4px 15px rgba(0,0,0,0.4);
            ">
            """, unsafe_allow_html=True)

            st.image(Image.open(img_path), use_column_width=True)

            st.markdown("</div>", unsafe_allow_html=True)

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

            # Severity Box
            if confidence > 85:
                color = "red"
                text = "🔴 HIGH SEVERITY INFECTION"
            elif confidence > 60:
                color = "orange"
                text = "🟠 MODERATE INFECTION LEVEL"
            else:
                color = "green"
                text = "🟢 LOW INFECTION LEVEL"

            st.markdown(f"""
            <div style='background:{color};
                        padding:15px;
                        border-radius:10px;
                        text-align:center;
                        color:white;
                        font-size:22px;
                        font-weight:bold;'>
            {text}
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

            pdf = generate_pdf(disease, confidence, info)

            st.download_button(
                "📄 Download Report as PDF",
                data=pdf,
                file_name=f"{disease}.pdf",
                mime="application/pdf"
            )
