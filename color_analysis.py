import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from PIL import Image
import os

# =========================
# CONFIG
# =========================

images_csv = "metadata/images.csv"
games_csv = "metadata/games.csv"
output_dir = "output"

palette_size = 5

os.makedirs(output_dir, exist_ok=True)

# =========================
# LOAD DATA
# =========================

df = pd.read_csv(images_csv)
games_df = pd.read_csv(games_csv)

# =========================
# HEX → RGB
# =========================

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

# =========================
# COMPUTE GAME PALETTES
# =========================

results = []

for game_id in df["game_id"].unique():
    game_frames = df[df["game_id"] == game_id]

    # ---------------------------------
    # COLLECT ALL DOMINANT COLORS
    # ---------------------------------

    all_colors = []

    for i in range(1, palette_size + 1):
        col_name = f"dom_color_{i}"

        if col_name in game_frames.columns:
            colors = game_frames[col_name].dropna().tolist()
            all_colors.extend(colors)

    # Count frequencies
    color_counts = Counter(all_colors)

    # Top colors
    top_colors = color_counts.most_common(10)

    # ---------------------------------
    # COMPUTE AVERAGE RGB
    # ---------------------------------

    rgb_values = []

    for color, count in top_colors:
        rgb = hex_to_rgb(color)

        # Weight by frequency
        for _ in range(count):
            rgb_values.append(rgb)

    avg_rgb = np.mean(rgb_values, axis=0).astype(int)

    results.append({
        "game_id": game_id,
        "avg_rgb": avg_rgb,
        "top_colors": top_colors
    })

# =========================
# VISUALIZATION 1
# AVERAGE COLORS
# =========================

fig, axes = plt.subplots(len(results), 1, figsize=(8, 2 * len(results)))

if len(results) == 1:
    axes = [axes]

for ax, result in zip(axes, results):
    rgb = result["avg_rgb"]

    color_img = np.zeros((100, 400, 3), dtype=np.uint8)
    color_img[:] = rgb

    ax.imshow(color_img)
    ax.set_title(result["game_id"], fontsize=14)
    ax.axis("off")

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "average_game_colors.png"), dpi=300)
plt.close()

print("Saved: average_game_colors.png")

# =========================
# VISUALIZATION 2
# PALETTE STRIPS
# =========================

fig, axes = plt.subplots(len(results), 1, figsize=(12, 2 * len(results)))

if len(results) == 1:
    axes = [axes]

for ax, result in zip(axes, results):
    top_colors = result["top_colors"]

    total = sum(count for _, count in top_colors)

    current_x = 0

    for color, count in top_colors:
        width = count / total

        ax.barh(
            y=0,
            width=width,
            left=current_x,
            color=color,
            height=1
        )

        current_x += width

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 0.5)
    ax.set_title(result["game_id"], fontsize=14)
    ax.axis("off")

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "game_palettes.png"), dpi=300)
plt.close()

print("Saved: game_palettes.png")

# =========================
# OPTIONAL:
# EXPORT PALETTE SUMMARY CSV
# =========================

summary_rows = []

for result in results:
    row = {
        "game_id": result["game_id"],
        "avg_rgb": str(tuple(result["avg_rgb"]))
    }

    for i, (color, count) in enumerate(result["top_colors"][:10]):
        row[f"color_{i+1}"] = color
        row[f"count_{i+1}"] = count

    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(
    os.path.join(output_dir, "palette_summary.csv"),
    index=False
)

print("Saved: palette_summary.csv")

print("\nPalette analysis complete.")