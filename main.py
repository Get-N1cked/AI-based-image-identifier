import cv2 #basically opencv
import numpy as np
import streamlit as st
from tensorflow.keras.applications.mobilenet_v2 import(
    MobileNetV2,
    preprocess_input,
    decode_predictions
)
from PIL import Image

def load_model():
    model = MobileNetV2(weights="imagenet")
    return model

def preprocess_image(image): #converts the image to the right dimensions before sending it to the model
    img = np.array(image)
    img = cv2.resize(img, (224, 224))# resizes the image
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img

def classify_image(model, image):
    try:
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image)
        decoded_predictions = decode_predictions(predictions, top=5)[0]
        return decoded_predictions
    except Exception as e:
        st.error(f"Error in proccessing the image: {str(e)}")
        return None

def main():
    st.set_page_config(page_title="Image Identifier 📷", page_icon="📷")
    st.title("Image Identifier")
    st.write("Upload the image to have it identified")
    @st.cache_resource
    def load_cached_model():
        return load_model()
    model = load_cached_model()

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = st.image(
            uploaded_file, caption="Uploaded Image", use_container_width=True
        )
        btn = st.button("Identify")
        if btn:
            with st.spinner("Identifying the image..."):
                image = Image.open(uploaded_file)
                predictions = classify_image(model, image)
                if predictions:
                    st.subheader("prediction")
                    for _, label, score in predictions:
                         st.write(f"**{label}**: {score:.2%}")
if __name__ == "__main__":
    main()



