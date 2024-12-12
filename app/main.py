from flask import Flask, request, jsonify
import os
from api.outfit_prediction import predict_outfit
from api.outfit_recommendation import match_outfits
from api.database import get_image_data_from_db
import pandas as pd

app = Flask(__name__)

@app.route('/predict_outfit', methods=['POST'])
def predict_outfit_api():
    data = request.get_json()
    catalog_id_upper = data.get("catalog_id_upper")
    catalog_id_bottom = data.get("catalog_id_bottom")

    # Fetch image URLs from database
    upper_image_data = get_image_data_from_db(catalog_id_upper)
    bottom_image_data = get_image_data_from_db(catalog_id_bottom)

    if upper_image_data and bottom_image_data:
        upper_image_url = upper_image_data['image_url']
        bottom_image_url = bottom_image_data['image_url']

        # Predict compatibility
        result = predict_outfit(upper_image_url, bottom_image_url)
        return jsonify({"result": result}), 200
    else:
        return jsonify({"error": "Invalid catalog IDs"}), 400


@app.route('/recommend_outfits', methods=['POST'])
def recommend_outfits_api():
    data = request.get_json()
    catalog_ids = data.get("catalog_ids")

    # Fetch images and categories from the database
    input_image_paths = []
    input_categories = []

    for catalog_id in catalog_ids:
        image_data = get_image_data_from_db(catalog_id)
        if image_data:
            input_image_paths.append(image_data['image_url'])
            input_categories.append(image_data['category'])

    # Load the annotations DataFrame (replace with actual CSV or database call)
    annotations_df = pd.read_csv('/path/to/annotations.csv')

    # Recommend outfits
    matching_outfits = match_outfits(input_image_paths, annotations_df, input_categories)
    return jsonify(matching_outfits), 200


if __name__ == '__main__':
    app.run(debug=True)
