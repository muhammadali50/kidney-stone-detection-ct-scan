import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

model = tf.keras.models.load_model("kidney_stone_detection_model.h5")

st.title("Kidney Stone Detection System")

# Upload CT scan image
uploaded_file = st.file_uploader("Upload CT Scan Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Read image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded CT Scan", use_column_width=True)

    # Convert to array
    img_array = np.array(image)

    # Resize
    img_array = cv2.resize(img_array, (150, 150))

    # Normalize
    img_array = img_array / 255.0

    # Expand dims
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)

    # Output handling
    if prediction[0][0] > 0.5:
        st.error("Kidney Stone Detected ❌")
    else:
        st.success("No Kidney Stone Detected ✅")