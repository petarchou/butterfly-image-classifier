import json
import os
import subprocess
import sys

import streamlit as st
import tensorflow as tf
from PIL import Image

from src.inference import predict_image

ARTIFACTS_DIR = "artifacts"
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "butterfly_model.keras")
CLASS_NAMES_PATH = os.path.join(ARTIFACTS_DIR, "class_names.json")
TRAIN_CONFIG_PATH = os.path.join(ARTIFACTS_DIR, "train_config.json")


@st.cache_resource
def load_resources():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(CLASS_NAMES_PATH):
        with st.spinner("Model not found — training now (this may take several minutes)..."):
            result = subprocess.run(
                [sys.executable, "train.py"],
                capture_output=True,
                text=True,
            )
        if result.returncode != 0:
            raise RuntimeError(f"Training failed:\n{result.stderr}")

    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"preprocess_input": tf.keras.applications.mobilenet_v3.preprocess_input},
    )
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    img_size = (224, 224)
    if os.path.exists(TRAIN_CONFIG_PATH):
        with open(TRAIN_CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        if "img_size" in cfg and len(cfg["img_size"]) == 2:
            img_size = tuple(cfg["img_size"])
    return model, class_names, img_size


def main():
    st.set_page_config(page_title="Butterfly Classifier", page_icon=":butterfly:")
    st.title("Butterfly Image Classifier")
    st.caption("Upload an image and get a species prediction from the trained model.")

    try:
        model, class_names, img_size = load_resources()
    except Exception as e:
        st.error(f"Could not load model artifacts: {e}")
        st.stop()

    uploaded_file = st.file_uploader(
        "Choose a butterfly image",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_file is None:
        st.info("Waiting for an image upload.")
        return

    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded image", use_container_width=True)
    except Exception as e:
        st.error(f"Invalid image file: {e}")
        return

    with st.spinner("Analyzing image..."):
        try:
            result = predict_image(
                model=model,
                image=image,
                class_names=class_names,
                img_size=img_size,
                top_k=3,
            )
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            return

    st.success(f"Prediction: {result['label']}")
    st.info(f"Confidence: {result['confidence']:.2f}%")
    st.progress(min(100, max(0, int(result["confidence"]))))

    st.subheader("Top 3 classes")
    for item in result["top_k"]:
        st.write(f"- {item['label']}: {item['confidence']:.2f}%")


if __name__ == "__main__":
    main()
