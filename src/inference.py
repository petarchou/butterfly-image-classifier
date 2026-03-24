from io import BytesIO
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from PIL import Image


def preprocess_pil_image(image: Image.Image, img_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    if image.mode != "RGB":
        image = image.convert("RGB")
    resized = image.resize(img_size)
    img_array = tf.keras.utils.img_to_array(resized)
    img_batch = np.expand_dims(img_array, axis=0)
    return img_batch


def preprocess_uploaded_bytes(file_bytes: bytes, img_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    image = Image.open(BytesIO(file_bytes))
    return preprocess_pil_image(image, img_size=img_size)


def predict_image(
    model: tf.keras.Model,
    image: Image.Image,
    class_names: List[str],
    img_size: Tuple[int, int] = (224, 224),
    top_k: int = 3,
):
    inputs = preprocess_pil_image(image, img_size=img_size)
    predictions = model.predict(inputs, verbose=0)[0]

    top_indices = np.argsort(predictions)[::-1][:top_k]
    top_results = [
        {
            "label": class_names[int(idx)],
            "confidence": float(predictions[int(idx)] * 100.0),
        }
        for idx in top_indices
    ]

    best = top_results[0]
    return {
        "label": best["label"],
        "confidence": best["confidence"],
        "top_k": top_results,
    }
