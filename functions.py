import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy

# Load the pre-trained MobileNetV2 model
model = tf.keras.applications.MobileNetV2(weights='imagenet')

# Function to preprocess and classify the image
def classify_image(image):
    image = tf.image.resize(image, (224, 224))
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    predictions = model.predict(tf.expand_dims(image, axis=0))
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions)
    return decoded_predictions[0][0][1]

# Streamlit UI
st.title("Dog and Cat Classifier")
st.write("Upload an image, and I'll tell you whether it's a dog or a cat!")

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Perform classification
    class_name = classify_image(image)
    st.write(f"Prediction: {class_name}")

