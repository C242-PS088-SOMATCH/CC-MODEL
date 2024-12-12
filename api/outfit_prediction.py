from sklearn.cluster import KMeans
from collections import Counter
from PIL import Image
import pandas as pd
import numpy as np

def extract_style_features(image_path):
    # Your style feature extraction logic here (using the style model)
    pass

def extract_dominant_color(image_path, k=5):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((150, 150))
    img_array = np.array(image)
    img_array = img_array.reshape((-1, 3))

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(img_array)
    colors = kmeans.cluster_centers_
    counts = Counter(kmeans.labels_)

    sorted_colors = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    dominant_color = colors[sorted_colors[0][0]]
    return dominant_color

def match_outfits(input_image_paths, annotations_df, input_categories):
    matching_outfits = {}
    input_features = []

    for img_path in input_image_paths:
        style_feature = extract_style_features(img_path)
        dominant_color = extract_dominant_color(img_path)
        input_features.append((style_feature, dominant_color))

    for _, row in annotations_df.iterrows():
        if row['category'] not in input_categories:
            for feature in input_features:
                if row['style'] == feature[0] and row['color_group'] == feature[1]:
                    if row['category'] not in matching_outfits:
                        matching_outfits[row['category']] = []
                    matching_outfits[row['category']].append(row.to_dict())
    return matching_outfits
