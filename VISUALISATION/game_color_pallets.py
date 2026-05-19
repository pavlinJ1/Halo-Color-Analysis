import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import matplotlib.patches as patches


images_path = "../metadata/images.csv"
games_path = "../metadata/games.csv"
output_path = "output/game_color_palettes.png"

os.makedirs("output", exist_ok=True)

df = pd.read_csv(images_path)
games = pd.read_csv(games_path)

# Preserve release/order from games.csv
games["game_order"] = range(len(games))

df = df.merge(
    games[["game_id", "title", "game_order"]],
    on="game_id",
    how="left"
)

games_ordered = games.sort_values("game_order")

k_palette = 5


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


def extract_game_palette(sub_df, prefix, k=5):
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

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

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


palette_rows = []

for _, game_row in games_ordered.iterrows():
    game_id = game_row["game_id"]
    title = game_row["title"]

    sub = df[df["game_id"] == game_id]

    if sub.empty:
        continue

    full_hexes, full_pcts = extract_game_palette(sub, "full", k_palette)
    nodark_hexes, nodark_pcts = extract_game_palette(sub, "nodark", k_palette)

    palette_rows.append({
        "title": title,
        "type": "Full palette",
        "colors": full_hexes,
        "weights": full_pcts
    })

    palette_rows.append({
        "title": title,
        "type": "No-dark palette",
        "colors": nodark_hexes,
        "weights": nodark_pcts
    })


n_rows = len(palette_rows)

fig, ax = plt.subplots(figsize=(13, n_rows * 0.45 + 1.5))

ax.set_xlim(0, 1)
ax.set_ylim(0, n_rows)
ax.axis("off")

for row_idx, row in enumerate(palette_rows):
    y = n_rows - row_idx - 1

    title_text = row["title"] if row["type"] == "Full palette" else ""
    type_text = row["type"]

    ax.text(
        -0.02,
        y + 0.5,
        title_text,
        ha="right",
        va="center",
        fontsize=9
    )

    ax.text(
        0.01,
        y + 0.5,
        type_text,
        ha="left",
        va="center",
        fontsize=8
    )

    x_start = 0.18
    available_width = 0.78
    x = x_start

    for color, weight in zip(row["colors"], row["weights"]):
        width = available_width * weight

        rect = patches.Rectangle(
            (x, y + 0.15),
            width,
            0.7,
            facecolor=color,
            edgecolor="white",
            linewidth=0.5
        )

        ax.add_patch(rect)
        x += width

    # If weights do not fill the full width because of missing colors
    if x < x_start + available_width:
        rect = patches.Rectangle(
            (x, y + 0.15),
            x_start + available_width - x,
            0.7,
            facecolor="#eeeeee",
            edgecolor="white",
            linewidth=0.5
        )
        ax.add_patch(rect)

ax.set_title("Dominant Color Palettes by Game", fontsize=14, pad=15)

plt.tight_layout()
plt.savefig(output_path, dpi=150, bbox_inches="tight")
plt.close()

print(f"Saved: {output_path}")