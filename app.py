from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import tensorflow as tf
import requests
from PIL import Image
import io
import numpy as np

# Load the model
model = load_model("compatible_fashion_v1.h5")

app = Flask(__name__)

def prepare_image_from_url(image_url):
    # Download the image from the URL
    response = requests.get(image_url)
    image = Image.open(io.BytesIO(response.content))
    image = image.convert('RGB')
    image = np.array(image)
    image = tf.image.resize(image, (224, 224)) / 255.0
    return image

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get image URLs from the request
        data = request.get_json()
        image_url_upper = data.get('upper')
        image_url_bottom = data.get('bottom')

        if not image_url_upper or not image_url_bottom:
            return jsonify({"error": "Both 'upper' and 'bottom' image URLs are required"}), 400

        # Prepare the images
        image_upper = prepare_image_from_url(image_url_upper)
        image_bottom = prepare_image_from_url(image_url_bottom)

        input_tensor = {
            "image1_input": tf.expand_dims(image_upper, axis=0),
            "image2_input": tf.expand_dims(image_bottom, axis=0)
        }
        
        # Predict using the model
        predictions = model.predict(input_tensor)
        predicted_class = 1 if predictions[0][0] >= 0.5 else 0
        
        # Send response
        if predicted_class == 1:
            return jsonify({"result": "Compatible"})
        else:
            return jsonify({"result": "Not Compatible"})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)