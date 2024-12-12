from flask import Flask, request, jsonify
import os
from api.outfit_prediction import get_outfit_prediction
from api.outfit_recommendation import match_outfits_with_color_and_style
from api.database import get_image_data_from_db
import pandas as pd

app = Flask(__name__)

@app.route('/predict_outfit', methods=['POST'])
def predict_outfit():
    return get_outfit_prediction()

@app.route('/recommend_outfits', methods=['POST'])
def recommend_outfits():
    return match_outfits_with_color_and_style()


if __name__ == '__main__':
    app.run(debug=True, port=os.getenv('APP_PORT'))
