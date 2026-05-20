import os
import colorsys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.cluster import KMeans


images_path = "../metadata/images.csv"
games_path = "../metadata/games.csv"
output_path = "output/color_features_overall.png"

os.makedirs("output", exist_ok=True)

df = pd.read_csv(images_path)
games = pd.read_csv(games_path)

df = df.merge(
    games[["game_id", "title"]],
    on="game_id",
    how="left"
)


def hex_to_rgb(hex_color):
    if pd.isna(hex_color) or str(hex_color).strip() == "":
        return None

    hex_color = str(hex_color).strip().lstrip("#")

    if len(hex_color) != 6:
        return None

    try:
        return np.array([
            int(hex_color[0:2], 16),
            int(hex_color[2:4], 16),
            int(hex_color[4:6], 16)
        ], dtype=float)
    except ValueError:
        return None


def rgb_to_hex(rgb):
    rgb = np.clip(rgb, 0, 255).astype(int)
    return "#{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])


def get_weighted_average_color(sub_df, prefix="nodark"):
    total_rgb = np.zeros(3)
    total_weight = 0

    for _, row in sub_df.iterrows():
        for i in range(1, 6):
            color = row.get(f"{prefix}_color_{i}")
            pct = row.get(f"{prefix}_color_{i}_pct")

            rgb = hex_to_rgb(color)

            if rgb is None or pd.isna(pct) or pct <= 0:
                continue

            total_rgb += rgb * float(pct)
            total_weight += float(pct)

    if total_weight == 0:
        return np.array([0, 0, 0], dtype=float)

    return total_rgb / total_weight


def extract_palette(sub_df, prefix="nodark", k=5):
    colors = []
    weights = []

    for _, row in sub_df.iterrows():
        for i in range(1, 6):
            color = row.get(f"{prefix}_color_{i}")
            pct = row.get(f"{prefix}_color_{i}_pct")

            rgb = hex_to_rgb(color)

            if rgb is None or pd.isna(pct) or pct <= 0:
                continue

            colors.append(rgb)
            weights.append(float(pct))

    if len(colors) == 0:
        return ["#000000"] * k, [0] * k

    colors = np.array(colors)
    weights = np.array(weights)

    if len(colors) < k:
        hexes = [rgb_to_hex(c) for c in colors]
        pcts = list(weights / weights.sum())

        while len(hexes) < k:
            hexes.append("#000000")
            pcts.append(0)

        return hexes, pcts

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(colors, sample_weight=weights)
    centers = kmeans.cluster_centers_

    cluster_weights = np.zeros(k)
    for label, weight in zip(labels, weights):
        cluster_weights[label] += weight

    order = np.argsort(cluster_weights)[::-1]
    centers = centers[order]
    cluster_weights = cluster_weights[order]
    cluster_weights = cluster_weights / cluster_weights.sum()

    hexes = [rgb_to_hex(c) for c in centers]
    pcts = cluster_weights.tolist()

    return hexes, pcts


def luminance(rgb_01):
    r, g, b = rgb_01
    return 0.299 * r + 0.587 * g + 0.114 * b


def text_color_for_bg(rgb_01):
    return "black" if luminance(rgb_01) > 0.58 else "white"


def brightness_color(value):
    v = np.clip(value / 255.0, 0, 1)
    return (v, v, v)


def saturation_color(value):
    s = np.clip(value / 255.0, 0, 1)
    h = 0.58
    v = 0.92
    return colorsys.hsv_to_rgb(h, s, v)


def contrast_color(value):
    v = np.clip(value / 255.0, 0, 1)
    return (v, v, v)


def warmth_color(value):
    value = np.clip(value, 0, 1)
    cool = np.array([55, 105, 170]) / 255.0
    warm = np.array([220, 120, 55]) / 255.0
    rgb = cool * (1 - value) + warm * value
    return tuple(rgb)


# Overall series metrics
avg_color_full_rgb = get_weighted_average_color(df, prefix="full")

overall_full_hexes, overall_full_pcts = extract_palette(df, prefix="full", k=5)
overall_nodark_hexes, overall_nodark_pcts = extract_palette(df, prefix="nodark", k=5)

avg_brightness = df["avg_brightness"].mean()
avg_saturation = df["avg_saturation"].mean()
avg_contrast = df["contrast"].mean()
avg_warmth = df["warmth"].mean()

columns = [
    "Average color\n(full)",
    "Brightness",
    "Saturation",
    "Contrast",
    "Warmth"
]

fig, ax = plt.subplots(figsize=(14, 5.8))

ax.set_xlim(0, len(columns))
ax.set_ylim(0, 4.2)
ax.axis("off")

# Title
ax.set_title(
    "Overall Halo Series Color Features",
    fontsize=18,
    pad=22
)

# Palette strips
palette_total_width = 4.8
x_start = 1.0

# Full palette
ax.text(
    0.75,
    3.25,
    "Overall full palette",
    ha="right",
    va="center",
    fontsize=11,
    fontweight="bold"
)

x = x_start
for color, weight in zip(overall_full_hexes, overall_full_pcts):
    width = palette_total_width * weight
    rect = patches.Rectangle(
        (x, 3.0),
        width,
        0.35,
        facecolor=color,
        edgecolor="white",
        linewidth=1.2
    )
    ax.add_patch(rect)
    x += width

# No-dark palette
ax.text(
    0.75,
    2.6,
    "Overall no-dark palette",
    ha="right",
    va="center",
    fontsize=11,
    fontweight="bold"
)

x = x_start
for color, weight in zip(overall_nodark_hexes, overall_nodark_pcts):
    width = palette_total_width * weight
    rect = patches.Rectangle(
        (x, 2.35),
        width,
        0.35,
        facecolor=color,
        edgecolor="white",
        linewidth=1.2
    )
    ax.add_patch(rect)
    x += width

# Column headers
for col_idx, col_name in enumerate(columns):
    ax.text(
        col_idx + 0.5,
        1.65,
        col_name,
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold"
    )

# Row label
ax.text(
    -0.15,
    0.75,
    "All Halo Games",
    ha="right",
    va="center",
    fontsize=11,
    fontweight="bold"
)

# Data cells
cell_data = [
    {
        "color": tuple(avg_color_full_rgb / 255.0),
        "label": rgb_to_hex(avg_color_full_rgb)
    },
    {
        "color": brightness_color(avg_brightness),
        "label": f"{avg_brightness:.1f}"
    },
    {
        "color": saturation_color(avg_saturation),
        "label": f"{avg_saturation:.1f}"
    },
    {
        "color": contrast_color(avg_contrast),
        "label": f"{avg_contrast:.1f}"
    },
    {
        "color": warmth_color(avg_warmth),
        "label": f"{avg_warmth:.2f}"
    }
]

for col_idx, cell in enumerate(cell_data):
    bg = cell["color"]

    rect = patches.Rectangle(
        (col_idx + 0.03, 0.3),
        0.94,
        0.84,
        facecolor=bg,
        edgecolor="white",
        linewidth=1.5
    )
    ax.add_patch(rect)

    ax.text(
        col_idx + 0.5,
        0.72,
        cell["label"],
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color=text_color_for_bg(bg)
    )

# Subtle guide line
ax.plot([-0.02, len(columns)], [1.25, 1.25], color="#eaeaea", linewidth=1.0)


plt.tight_layout()
plt.savefig(output_path, dpi=200, bbox_inches="tight")
plt.close()

print(f"Saved: {output_path}")