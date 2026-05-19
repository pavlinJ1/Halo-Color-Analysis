import cv2
import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm
import os


csv_path = "../metadata/images.csv"
image_dir = "../images"

k_colors = 5      # number of dominant colors
resize_dim = 100  # scale for faster analysis


df = pd.read_csv(csv_path)


df["avg_hue"] = np.nan
df["avg_saturation"] = np.nan
df["avg_brightness"] = np.nan

for i in range(k_colors):
    col = f"dom_color_{i+1}"
    df[col] = ""



def get_avg_hsv(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    return np.mean(h), np.mean(s), np.mean(v)

def get_dominant_colors(image, k=5):
    image_small = cv2.resize(image, (resize_dim, resize_dim))
    pixels = image_small.reshape(-1, 3)

    kmeans = MiniBatchKMeans(n_clusters=k, batch_size=1024, n_init=3)
    kmeans.fit(pixels)

    colors = kmeans.cluster_centers_.astype(int)

    hex_colors = []
    for c in colors:
        b, g, r = c
        hex_colors.append('#{:02x}{:02x}{:02x}'.format(r, g, b))

    return hex_colors



for i in tqdm(range(len(df)), desc="Processing images"):
    path = os.path.join(image_dir, df.at[i, "filename"])

    if not os.path.exists(path):
        print(f"Missing: {path}")
        continue

    image = cv2.imread(path)

    if image is None:
        print(f"Failed to load: {path}")
        continue

    avg_h, avg_s, avg_v = get_avg_hsv(image)
    dom_colors = get_dominant_colors(image, k_colors)

    df.loc[i, "avg_hue"] = round(avg_h, 2)
    df.loc[i, "avg_saturation"] = round(avg_s, 2)
    df.loc[i, "avg_brightness"] = round(avg_v, 2)

    for j in range(k_colors):
        df.loc[i, f"dom_color_{j+1}"] = dom_colors[j]


df.to_csv(csv_path, index=False)

print("\nAnalysis complete.")

