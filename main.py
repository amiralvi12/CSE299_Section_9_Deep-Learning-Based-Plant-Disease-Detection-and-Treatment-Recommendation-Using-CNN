import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st


st.set_page_config(
    page_title="Plant Disease Classifier",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="collapsed"
)


st.markdown("""
    <style>
        .main {
            background-color: #f4f6f9;
        }
        .title {
            font-size: 3em;
            font-weight: bold;
            color: #2e7d32;
            text-align: center;
            margin-bottom: 10px;
        }
        .footer {
            text-align: center;
            color: grey;
            font-size: 0.9em;
            margin-top: 2rem;
        }
        .prediction {
            font-size: 1.5em;
            font-weight: bold;
            color: #1565c0;
        }
    </style>
""", unsafe_allow_html=True)


st.markdown('<div class="title">🌱 Plant Disease Classifier 🌱</div>', unsafe_allow_html=True)


working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/trained_model/best_model_epoch_49.h5"
model = tf.keras.models.load_model(model_path)
class_indices = json.load(open(f"{working_dir}/class_indices.json"))



def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path).resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0).astype('float32') / 255.
    return img_array



def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_index = np.argmax(predictions, axis=1)[0]
    return class_indices[str(predicted_index)]



uploaded_image = st.file_uploader("📤 Upload an image of a plant leaf", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(image.resize((200, 200)), caption="Uploaded Leaf Image", use_column_width=False)

    with col2:
        if st.button('🔍 Classify'):
            prediction = predict_image_class(model, uploaded_image, class_indices)
            st.markdown(f'<div class="prediction">🩺 Prediction: <span>{prediction}</span></div>', unsafe_allow_html=True)


st.markdown('<div class="footer">Developed with  using Streamlit</div>', unsafe_allow_html=True)
