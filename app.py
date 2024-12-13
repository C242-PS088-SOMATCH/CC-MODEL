from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import tensorflow as tf

app = Flask(__name__)

# Path model di GCS
MODEL_PATH = "gs://somatch/model/compatible_fashion_v1.h5"

# Variabel global untuk menyimpan model
model = None

def load_model_from_gcs():
    global model
    print("Loading model from GCS...")
    model = load_model(MODEL_PATH)
    print("Model successfully loaded!")

@app.before_first_request
def initialize_model():
    # Muat model hanya sekali saat API pertama kali dipanggil
    load_model_from_gcs()

def prepare_input(image_path_upper, image_path_bottom):
    # Memproses gambar untuk prediksi
    image_upper = tf.io.read_file(image_path_upper)
    image_upper = tf.image.decode_jpeg(image_upper, channels=3)
    image_upper = tf.image.resize(image_upper, (224, 224))
    image_upper = image_upper / 255.0

    image_bottom = tf.io.read_file(image_path_bottom)
    image_bottom = tf.image.decode_jpeg(image_bottom, channels=3)
    image_bottom = tf.image.resize(image_bottom, (224, 224))
    image_bottom = image_bottom / 255.0

    return {
        "image1_input": tf.expand_dims(image_upper, axis=0),
        "image2_input": tf.expand_dims(image_bottom, axis=0),
    }

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    image_path_upper = data.get("image_path_upper")
    image_path_bottom = data.get("image_path_bottom")
    
    if not image_path_upper or not image_path_bottom:
        return jsonify({"error": "Missing image paths"}), 400

    # Persiapkan input
    input_tensor = prepare_input(image_path_upper, image_path_bottom)

    # Lakukan prediksi
    predictions = model.predict(input_tensor)
    predicted_class = 1 if predictions[0][0] >= 0.5 else 0

    return jsonify({
        "predictions": predictions.tolist(),
        "class": "Compatible" if predicted_class == 1 else "Not Compatible"
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
