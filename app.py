import streamlit as st
import tensorflow as tf
import cv2
import numpy as np

# Load the saved model
model = tf.keras.models.load_model("counterfeit_detection_model.h5")

# Configure the app
st.title("QR Code Authenticity Classifier")
st.write("Upload a QR code image to check if it's original or counterfeit.")

# File upload widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Preprocess the image
    image = cv2.imdecode(
        np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR
    )
    resized_img = cv2.resize(image, (224, 224)) / 255.0
    input_img = np.expand_dims(resized_img, axis=0)

    # Predict
    prediction = model.predict(input_img)
    result = (
        "Counterfeit (Second Print)" if prediction > 0.5 else "Original (First Print)"
    )

    # Display results
    st.image(image, caption="Uploaded QR Code", use_column_width=True)
    st.subheader(f"Prediction: **{result}**")
    st.write(f"Confidence: {prediction[0][0]:.2f}")
