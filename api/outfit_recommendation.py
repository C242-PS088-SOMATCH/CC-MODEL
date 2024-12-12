import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from collections import Counter
from PIL import Image
import api
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import os

# Load pre-trained style model
style_model = tf.keras.models.load_model(os.getenv(MODEL_PATH_STYLE))

class_names = ['Athleisure', 'Business Casual', 'Formal', 'Gothic', 'Minimalist', 'Preppy', 'Punk', 'Streetwear', 'Vintage']

def extract_style_features(image_path):
    """
    Extracts style features from an image using a pre-trained model.

    Args:
        image_path (str): Path to the image file.

    Returns:
        np.ndarray: Extracted style feature vector.
    """
    image = Image.open(image_path).convert("RGB").resize((224, 224))
    img_array = np.array(image) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Tambah dimensi batch

    predictions = style_model.predict(img_array)
    predicted_label_index = np.argmax(predictions)
    predicted_style = class_names[predicted_label_index]
    return predicted_style

def extract_dominant_color(image_path, k=5):
    """
    Extracts the dominant color from an image using K-Means clustering.

    Args:
        image_path (str): Path to the image file.
        k (int): Number of clusters for K-Means.

    Returns:
        np.ndarray: Dominant color as an RGB array.
    """
    # Open the image and convert to RGB
    image = Image.open(image_path).convert("RGB")
    image = image.resize((150, 150))  # Resize gambar
    img_array = np.array(image)
    img_array = img_array.reshape((-1, 3))  # Reshape ke [n_pixels, 3]

    # Cluster colors using K-Means
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(img_array)
    colors = kmeans.cluster_centers_
    counts = Counter(kmeans.labels_)

    # Sort colors based on frequency
    sorted_colors = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    dominant_color = colors[sorted_colors[0][0]]
    return dominant_color

def rgb_to_hsv(rgb):
    """
    Converts an RGB color to HSV.

    Args:
        rgb (np.ndarray): RGB color array.

    Returns:
        tuple: HSV representation.
    """
    return mcolors.rgb_to_hsv(rgb / 255.0)

def assign_color_group(dominant_color):
    """
    Assigns a color group based on HSV values.

    Args:
        dominant_color (np.ndarray): Dominant RGB color.

    Returns:
        str: Color group name.
    """
    # Convert color to HSV
    dominant_hsv = rgb_to_hsv(dominant_color)

    h, s, v = dominant_hsv
    if s < 0.2 and v > 0.8:
        return "Neutral"
    elif s < 0.3 and v < 0.5:
        return "Dark"
    elif s < 0.4:
        return "Pastel"
    elif 0.1 <= h <= 0.2 and s > 0.3:
        return "Earth Tone"
    elif s > 0.7 and v > 0.5:
        return "Bright/Vivid"
    elif 0.9 <= h <= 1.0 or 0.0 <= h <= 0.1:
        return "Warm"
    elif 0.5 <= h <= 0.75:
        return "Cool"
    else:
        return "Other"

def match_outfits_with_color_and_style(body):
    """
    Matches outfits based on both color group and style similarity and selects a random outfit for each category.

    Args:
        body: JSON object containing input categories and their IDs.

    Returns:
        dict: Dictionary containing input_images and recommended_outfits.
    """
    input_images = {key: value for key, value in body.items() if value}
    input_features = []
    for category_id in input_images.values():
        data = api.database.get_image_data_from_db(category_id)
        if data:
            style_feature = extract_style_features(data['image_url'])
            dominant_color = extract_dominant_color(data['image_url'])
            color_group = assign_color_group(dominant_color)
            set_feature_image(data['id'], dominant_color, style_feature, color_group, f"{data['category']} {style_feature} {color_group}") 
            input_features.append((style_feature, color_group))

    matching_outfits = {}
    annotations_df = api.database.get_all_image_data_from_db()

    for _, row in annotations_df.iterrows():
        row_category = row['category']
        row_style = row['style']
        row_color_group = row['color_group']

        if row_category in input_images:
            continue

        for predicted_style, color_group in input_features:
            if row_style == predicted_style and row_color_group == color_group:
                if row_category not in matching_outfits:
                    matching_outfits[row_category] = []
                matching_outfits[row_category].append(row)
                break

    recommended_outfits = {}
    for category in annotations_df['category'].unique():
        if category in input_images:
            continue
        outfits_of_type = annotations_df[annotations_df['category'] == category]
        if not outfits_of_type.empty:
            recommended_outfits[category] = outfits_of_type.sample(n=1).to_dict('records')[0]

    return {
        "message": "Recommendation is successful",
        "data": {
            "input_images": input_images,
            "recommended_outfits": recommended_outfits
        }
    }