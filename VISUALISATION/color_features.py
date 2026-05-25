import os
import colorsys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


images_path = "../metadata/images.csv"
games_path = "../metadata/games.csv"
output_path = "output/color_features.png"

os.makedirs("output", exist_ok=True)

df = pd.read_csv(images_path)
games = pd.read_csv(games_path)


games["game_order"] = range(len(games))

df = df.merge(
    games[["game_id", "title", "game_order"]],
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


summary_rows = []

for _, game_row in games.sort_values("game_order").iterrows():
    game_id = game_row["game_id"]
    title = game_row["title"]

    sub = df[df["game_id"] == game_id]

    if sub.empty:
        continue

    avg_color_rgb = get_weighted_average_color(sub, prefix="nodark")

    summary_rows.append({
        "title": title,
        "avg_color_rgb": avg_color_rgb,
        "avg_color_hex": rgb_to_hex(avg_color_rgb),
        "avg_brightness": sub["avg_brightness"].mean(),
        "avg_saturation": sub["avg_saturation"].mean(),
        "contrast": sub["contrast"].mean(),
        "warmth": sub["warmth"].mean()
    })

summary = pd.DataFrame(summary_rows)

columns = [
    "Average color",
    "Brightness",
    "Saturation",
    "Contrast",
    "Warmth"
]

n_games = len(summary)
n_cols = len(columns)

fig, ax = plt.subplots(figsize=(13, n_games * 0.65 + 2))

ax.set_xlim(0, n_cols)
ax.set_ylim(0, n_games)
ax.axis("off")

for col_idx, col_name in enumerate(columns):
    ax.text(
        col_idx + 0.5,
        n_games + 0.25,
        col_name,
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold"
    )

for row_idx, row in summary.iterrows():
    y = n_games - row_idx - 1

    ax.text(
        -0.15,
        y + 0.5,
        row["title"],
        ha="right",
        va="center",
        fontsize=10
    )

    cell_data = [
        {
            "value": row["avg_color_hex"],
            "color": tuple(row["avg_color_rgb"] / 255.0),
            "label": row["avg_color_hex"]
        },
        {
            "value": row["avg_brightness"],
            "color": brightness_color(row["avg_brightness"]),
            "label": f"{row['avg_brightness']:.1f}"
        },
        {
            "value": row["avg_saturation"],
            "color": saturation_color(row["avg_saturation"]),
            "label": f"{row['avg_saturation']:.1f}"
        },
        {
            "value": row["contrast"],
            "color": contrast_color(row["contrast"]),
            "label": f"{row['contrast']:.1f}"
        },
        {
            "value": row["warmth"],
            "color": warmth_color(row["warmth"]),
            "label": f"{row['warmth']:.2f}"
        }
    ]

    for col_idx, cell in enumerate(cell_data):
        bg = cell["color"]

        rect = patches.Rectangle(
            (col_idx + 0.03, y + 0.08),
            0.94,
            0.84,
            facecolor=bg,
            edgecolor="white",
            linewidth=1.5
        )
        ax.add_patch(rect)

        ax.text(
            col_idx + 0.5,
            y + 0.5,
            cell["label"],
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color=text_color_for_bg(bg)
        )

for y in range(n_games + 1):
    ax.plot(
        [-0.02, n_cols],
        [y, y],
        color="#eeeeee",
        linewidth=0.8,
        zorder=0
    )

ax.set_title(
    "Average Color Features\n",
    fontsize=17,
    pad=28
)

plt.tight_layout()
plt.savefig(output_path, dpi=200, bbox_inches="tight")
plt.close()

print(f"Saved: {output_path}")