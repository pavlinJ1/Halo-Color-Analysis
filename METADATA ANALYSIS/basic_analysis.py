import cv2
import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm
import os


csv_path = "../metadata/images.csv"
image_dir = "../images"

k_colors = 5
resize_dim = 120

df = pd.read_csv(csv_path)

for col in [
    "avg_hue",
    "avg_saturation",
    "avg_brightness",
    "brightness_std",
    "saturation_std",
    "contrast",
    "dark_pixel_pct"
]:
    df[col] = np.nan

for i in range(k_colors):
    df[f"full_color_{i+1}"] = ""
    df[f"full_color_{i+1}_pct"] = np.nan


for i in range(k_colors):
    df[f"nodark_color_{i+1}"] = ""
    df[f"nodark_color_{i+1}_pct"] = np.nan


def load_image_rgb(path):
    img = cv2.imread(path)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def get_hsv_stats(image_rgb):
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    hue_rad = h.astype(np.float32) * 2 * np.pi / 180
    mean_sin = np.mean(np.sin(hue_rad))
    mean_cos = np.mean(np.cos(hue_rad))
    avg_h = np.arctan2(mean_sin, mean_cos)

    if avg_h < 0:
        avg_h += 2 * np.pi

    avg_h = avg_h * 180 / (2 * np.pi)

    avg_s = np.mean(s)
    avg_v = np.mean(v)

    brightness_std = np.std(v)
    saturation_std = np.std(s)
    contrast = np.percentile(v, 95) - np.percentile(v, 5)

    dark_pixel_pct = np.mean(v < 25)

    return avg_h, avg_s, avg_v, brightness_std, saturation_std, contrast, dark_pixel_pct


def extract_palette_rgb(image_rgb, k=5, remove_dark=False, dark_threshold=25):
    image_small = cv2.resize(image_rgb, (resize_dim, resize_dim))
    pixels = image_small.reshape(-1, 3).astype(float)

    if remove_dark:
        hsv = cv2.cvtColor(image_small, cv2.COLOR_RGB2HSV)
        v = hsv[:, :, 2].reshape(-1)
        pixels = pixels[v >= dark_threshold]

    if len(pixels) == 0:
        return [""] * k, [np.nan] * k

    if len(pixels) < k:
        unique_colors = np.unique(pixels.astype(int), axis=0)
        colors = unique_colors[:k]
        pcts = [1 / len(colors)] * len(colors)

        hex_colors = [
            "#{:02x}{:02x}{:02x}".format(r, g, b)
            for r, g, b in colors
        ]

        while len(hex_colors) < k:
            hex_colors.append("")
            pcts.append(np.nan)

        return hex_colors, pcts

    kmeans = MiniBatchKMeans(
        n_clusters=k,
        batch_size=2048,
        n_init=10,
        random_state=42
    )

    labels = kmeans.fit_predict(pixels)
    centers = kmeans.cluster_centers_.astype(int)

    counts = np.bincount(labels, minlength=k)
    percentages = counts / counts.sum()

    order = np.argsort(percentages)[::-1]

    hex_colors = []
    sorted_pcts = []

    for idx in order:
        r, g, b = centers[idx]
        hex_colors.append("#{:02x}{:02x}{:02x}".format(r, g, b))
        sorted_pcts.append(round(float(percentages[idx]), 4))

    return hex_colors, sorted_pcts


for i in tqdm(range(len(df)), desc="Processing images"):
    path = os.path.join(image_dir, df.at[i, "filename"])

    if not os.path.exists(path):
        print(f"Missing: {path}")
        continue

    image = load_image_rgb(path)

    if image is None:
        print(f"Failed to load: {path}")
        continue

    stats = get_hsv_stats(image)

    df.loc[i, "avg_hue"] = round(stats[0], 2)
    df.loc[i, "avg_saturation"] = round(stats[1], 2)
    df.loc[i, "avg_brightness"] = round(stats[2], 2)
    df.loc[i, "brightness_std"] = round(stats[3], 2)
    df.loc[i, "saturation_std"] = round(stats[4], 2)
    df.loc[i, "contrast"] = round(stats[5], 2)
    df.loc[i, "dark_pixel_pct"] = round(stats[6], 4)

    full_colors, full_pcts = extract_palette_rgb(
        image,
        k=k_colors,
        remove_dark=False
    )

    nodark_colors, nodark_pcts = extract_palette_rgb(
        image,
        k=k_colors,
        remove_dark=True,
        dark_threshold=25
    )

    for j in range(k_colors):
        df.loc[i, f"full_color_{j+1}"] = full_colors[j]
        df.loc[i, f"full_color_{j+1}_pct"] = full_pcts[j]

        df.loc[i, f"nodark_color_{j+1}"] = nodark_colors[j]
        df.loc[i, f"nodark_color_{j+1}_pct"] = nodark_pcts[j]


df.to_csv(csv_path, index=False)

print("\nAnalysis complete.")