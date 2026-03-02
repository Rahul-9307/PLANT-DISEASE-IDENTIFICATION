import streamlit as st
import tensorflow as tf
import numpy as np
import os
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="AgriSens - Smart Plant Disease Detection",
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
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___healthy"
]

# -------------------------------------------------
# MULTILINGUAL DISEASE DATABASE
# -------------------------------------------------
disease_info = {
    "Tomato___Early_blight": {
        "English": {
            "description": "Fungal disease causing brown spots on leaves.",
            "cause": "Caused by Alternaria fungus in humid weather.",
            "prevention": "Avoid overhead watering and maintain spacing.",
            "treatment": "Spray Mancozeb fungicide every 7-10 days."
        },
        "Marathi": {
            "description": "ही बुरशीजन्य रोग आहे ज्यामुळे पानांवर तपकिरी डाग पडतात.",
            "cause": "ओलसर वातावरणात Alternaria बुरशीमुळे होतो.",
            "prevention": "वरून पाणी देणे टाळा आणि झाडांमध्ये अंतर ठेवा.",
            "treatment": "मॅन्कोझेब फवारणी ७-१० दिवसांनी करा."
        },
        "Hindi": {
            "description": "यह फंगल रोग है जिससे पत्तियों पर भूरे धब्बे बनते हैं।",
            "cause": "नमी वाले वातावरण में Alternaria फंगस के कारण होता है।",
            "prevention": "ऊपर से पानी न दें और पौधों में दूरी रखें।",
            "treatment": "मैनकोजेब का छिड़काव 7-10 दिन में करें।"
        }
    },

    "Tomato___Late_blight": {
        "English": {
            "description": "Serious fungal disease causing dark water-soaked lesions.",
            "cause": "High humidity and cool temperature.",
            "prevention": "Ensure proper drainage and air circulation.",
            "treatment": "Apply Metalaxyl fungicide immediately."
        },
        "Marathi": {
            "description": "गंभीर बुरशीजन्य रोग ज्यामुळे पाने काळी पडतात.",
            "cause": "जास्त आर्द्रता आणि थंड हवामान.",
            "prevention": "पाण्याचा निचरा व्यवस्थित ठेवा.",
            "treatment": "मेटालेक्सिल फवारणी त्वरित करा."
        },
        "Hindi": {
            "description": "गंभीर फंगल रोग जिससे पत्तियाँ काली पड़ जाती हैं।",
            "cause": "अधिक नमी और ठंडा मौसम।",
            "prevention": "पानी का उचित निकास रखें।",
            "treatment": "मेटालेक्सिल का तुरंत छिड़काव करें।"
        }
    },

    "Tomato___healthy": {
        "English": {
            "description": "The plant is healthy.",
            "cause": "No infection detected.",
            "prevention": "Maintain proper irrigation and fertilization.",
            "treatment": "No treatment required."
        },
        "Marathi": {
            "description": "वनस्पती निरोगी आहे.",
            "cause": "कोणताही रोग आढळला नाही.",
            "prevention": "योग्य पाणी आणि खत व्यवस्थापन ठेवा.",
            "treatment": "उपचाराची गरज नाही."
        },
        "Hindi": {
            "description": "पौधा स्वस्थ है।",
            "cause": "कोई संक्रमण नहीं मिला।",
            "prevention": "सही सिंचाई और खाद प्रबंधन रखें।",
            "treatment": "किसी उपचार की आवश्यकता नहीं।"
        }
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
def generate_pdf(disease, confidence, language):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)

    info = disease_info[disease][language]

    c.drawString(100, 750, "AgriSens - Plant Disease Report")
    c.drawString(100, 720, f"Disease: {disease}")
    c.drawString(100, 700, f"Confidence: {confidence:.2f}%")

    c.drawString(100, 670, f"Description: {info['description']}")
    c.drawString(100, 640, f"Cause: {info['cause']}")
    c.drawString(100, 610, f"Prevention: {info['prevention']}")
    c.drawString(100, 580, f"Treatment: {info['treatment']}")

    c.save()
    buffer.seek(0)
    return buffer

# -------------------------------------------------
# CENTER LAYOUT
# -------------------------------------------------
col1, col2, col3 = st.columns([1,3,1])

with col2:

    if os.path.exists(IMAGE_PATH):
        img = Image.open(IMAGE_PATH)
        st.image(img, use_column_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    language = st.selectbox("Select Language / भाषा निवडा / भाषा चुनें",
                            ["English", "Marathi", "Hindi"])

    st.markdown("<br>", unsafe_allow_html=True)

    title_text = {
        "English": "Disease Recognition",
        "Marathi": "रोग ओळख प्रणाली",
        "Hindi": "रोग पहचान प्रणाली"
    }

    st.markdown(
        f"<h1 style='text-align:center;color:#4CAF50;'>{title_text[language]}</h1>",
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    test_image = st.file_uploader("Upload Plant Leaf Image")

    if test_image is not None:
        st.image(test_image, use_column_width=True)

        if st.button("Predict"):
            result_index, confidence = model_prediction(test_image)
            disease_name = class_name[result_index]

            st.success(f"Prediction: {disease_name}")
            st.info(f"Confidence: {confidence:.2f}%")

            info = disease_info[disease_name][language]

            st.write("### Description")
            st.write(info["description"])

            st.write("### Cause")
            st.write(info["cause"])

            st.write("### Prevention")
            st.write(info["prevention"])

            st.write("### Treatment")
            st.write(info["treatment"])

            pdf = generate_pdf(disease_name, confidence, language)

            st.download_button(
                label="📄 Download Report as PDF",
                data=pdf,
                file_name="plant_disease_report.pdf",
                mime="application/pdf"
            )
