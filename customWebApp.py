import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array


def preprocess_image(image):
    image = tf.image.resize(image, (150, 150))
    image = img_to_array(image)
    image = image / 255.0
    image = tf.expand_dims(image, axis=0)
    return image

def decode_predictions(predictions):
    class_name = ""
    if predictions > 0.5:
        class_name = "Cat"
    else:
        class_name = "Dog"
    return class_name

# Load the custom model
custom_model = tf.keras.models.load_model('custom_model.keras')

# Function to preprocess and classify the image using the custom model
def classify_image(image):
    # Preprocess the image as required by your custom model
    image = preprocess_image(image)  # You need to implement this function
    # Perform classification using the custom model
    predictions = custom_model.predict(image)
    # Decode the predictions if needed
    class_name = decode_predictions(predictions)  # You need to implement this function
    return class_name



# Streamlit UI
st.title("Custom Image Classifier")
st.write("Upload an image, and I'll classify it!")

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Perform classification
    class_name = classify_image(image)
    st.write(f"Prediction: {class_name}")
