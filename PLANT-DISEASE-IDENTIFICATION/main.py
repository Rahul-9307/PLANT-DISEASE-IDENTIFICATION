import streamlit as st
import tensorflow as tf
import numpy as np
import os
from PIL import Image

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
# LOAD MODEL
# -------------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# -------------------------------------------------
# PREDICTION FUNCTION
# -------------------------------------------------
def model_prediction(test_image):
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    predictions = model.predict(input_arr)
    return np.argmax(predictions)

# -------------------------------------------------
# CENTER CONTENT
# -------------------------------------------------
col1, col2, col3 = st.columns([1,3,1])

with col2:

    # Banner
    if os.path.exists(IMAGE_PATH):
        img = Image.open(IMAGE_PATH)
        st.image(img, use_column_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Page selector
    app_mode = st.selectbox("Select a Page", ["HOME", "DISEASE RECOGNITION"])

    st.markdown("<br>", unsafe_allow_html=True)

    # HOME PAGE
    if app_mode == "HOME":
        st.markdown(
            "<h1 style='text-align: center; color: #2e8b57;'>SMART PLANT DISEASE DETECTION</h1>",
            unsafe_allow_html=True
        )
        st.markdown(
            "<p style='text-align: center;'>Upload a plant leaf image and our AI model will detect the disease.</p>",
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
                result_index = model_prediction(test_image)

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

                st.success(f"🌿 Prediction: {class_name[result_index]}")
