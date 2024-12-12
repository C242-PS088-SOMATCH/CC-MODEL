from tensorflow.keras.models import load_model
import tensorflow as tf
import os
from database import get_image_data_from_db
import json

# Load the .h5 model
model_path = os.getenv("MODEL_PATH_FASHION")
if not model_path:
    raise ValueError("MODEL_PATH_FASHION environment variable not set.")
    
model = load_model(model_path)
print("Model loaded from model.h5")

def prepare_input(image_path_upper, image_path_bottom):
    image = tf.io.read_file(image_path_upper)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (224, 224))
    image = image / 255.0

    image1 = tf.io.read_file(image_path_bottom)
    image1 = tf.image.decode_jpeg(image1, channels=3)
    image1 = tf.image.resize(image1, (224, 224))
    image1 = image1 / 255.0

    return {"image1_input": tf.expand_dims(image, axis=0), "image2_input": tf.expand_dims(image1, axis=0)}

def imagePath(item_id):
    data = get_image_data_from_db(item_id)
    if data:
        return data.get('image_url')  
    else:
        raise ValueError(f"Tidak ada data ditemukan untuk ID: {item_id}")

def get_outfit_prediction():
    try:
        body = request.get_json()
        input_tensor = prepare_input(imagePath(body.upperware), imagePath(body.bottomware))
        predictions = model.predict(input_tensor)
        predicted_class = 1 if predictions[0][0] >= 0.5 else 0

        if predicted_class == 1:
            return {
                "message": "Outfit prediction success",
                "prediction": "Compatible"
            }
        else:
            return {
                "message": "Outfit prediction success",
                "prediction": "Not Compatible"
            }

    except Exception as e:
        return {
            "message": "Error",
            "error": str(e)
        }, 500

