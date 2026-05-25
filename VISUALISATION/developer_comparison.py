import os
import colorsys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.cluster import KMeans


images_path = "../metadata/images.csv"
games_path = "../metadata/games.csv"
output_path = "output/developer_comparison.png"

os.makedirs("output", exist_ok=True)

df = pd.read_csv(images_path)
games = pd.read_csv(games_path)


developer_order = list(pd.unique(games["developer"]))

df = df.merge(
    games[["game_id", "title", "developer"]],
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


def luminance(rgb_01):
    r, g, b = rgb_01
    return 0.299 * r + 0.587 * g + 0.114 * b


def text_color_for_bg(rgb_01):
    return "black" if luminance(rgb_01) > 0.58 else "white"


def get_weighted_average_color(sub_df, prefix="full"):
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


def extract_palette(sub_df, prefix="full", k=5):
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

    return [rgb_to_hex(c) for c in centers], cluster_weights.tolist()


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


def metric_color(metric, value):
    if metric == "avg_brightness":
        return brightness_color(value)
    if metric == "avg_saturation":
        return saturation_color(value)
    if metric == "contrast":
        return contrast_color(value)
    if metric == "warmth":
        return warmth_color(value)
    if metric == "valence":
        t = np.clip((value + 1) / 2, 0, 1)
        neg = np.array([80, 65, 105]) / 255.0
        pos = np.array([220, 190, 105]) / 255.0
        return tuple(neg * (1 - t) + pos * t)
    if metric == "arousal":
        t = np.clip((value + 1) / 2, 0, 1)
        low = np.array([45, 80, 130]) / 255.0
        high = np.array([220, 80, 55]) / 255.0
        return tuple(low * (1 - t) + high * t)

    return (0.8, 0.8, 0.8)


def draw_palette_bar(ax, x, y, width, height, colors, weights):
    cur_x = x

    for color, weight in zip(colors, weights):
        segment_width = width * weight

        rect = patches.Rectangle(
            (cur_x, y),
            segment_width,
            height,
            facecolor=color,
            edgecolor="white",
            linewidth=0.8
        )

        ax.add_patch(rect)
        cur_x += segment_width


def draw_metric_cell(ax, x, y, width, height, label, value, metric):
    bg = metric_color(metric, value)

    rect = patches.Rectangle(
        (x, y),
        width,
        height,
        facecolor=bg,
        edgecolor="white",
        linewidth=1.2
    )

    ax.add_patch(rect)

    if metric in ["warmth", "valence", "arousal"]:
        value_label = f"{value:.2f}"
    else:
        value_label = f"{value:.1f}"

    ax.text(
        x + width / 2,
        y + height * 0.62,
        value_label,
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color=text_color_for_bg(bg)
    )

    ax.text(
        x + width / 2,
        y + height * 0.28,
        label,
        ha="center",
        va="center",
        fontsize=7.8,
        color=text_color_for_bg(bg)
    )


summary_rows = []

for developer in developer_order:
    sub = df[df["developer"] == developer]

    if sub.empty:
        continue

    game_titles = (
        games[games["developer"] == developer]["title"]
        .dropna()
        .tolist()
    )

    full_palette, full_weights = extract_palette(sub, prefix="full", k=5)
    nodark_palette, nodark_weights = extract_palette(sub, prefix="nodark", k=5)
    avg_color_rgb = get_weighted_average_color(sub, prefix="full")

    summary_rows.append({
        "developer": developer,
        "games": game_titles,
        "full_palette": full_palette,
        "full_weights": full_weights,
        "nodark_palette": nodark_palette,
        "nodark_weights": nodark_weights,
        "avg_color_rgb": avg_color_rgb,
        "avg_color_hex": rgb_to_hex(avg_color_rgb),
        "avg_brightness": sub["avg_brightness"].mean(),
        "avg_saturation": sub["avg_saturation"].mean(),
        "contrast": sub["contrast"].mean(),
        "warmth": sub["warmth"].mean(),
        "valence": sub["valence"].mean(),
        "arousal": sub["arousal"].mean(),
        "frame_count": len(sub)
    })


summary = pd.DataFrame(summary_rows)

metrics = [
    ("avg_brightness", "Brightness"),
    ("avg_saturation", "Saturation"),
    ("contrast", "Contrast"),
    ("warmth", "Warmth"),
    ("valence", "Pleasure"),
    ("arousal", "Arousal")
]

n_devs = len(summary)

fig, ax = plt.subplots(figsize=(15, 3.4 * n_devs))

ax.set_xlim(0, 1)
ax.set_ylim(0, n_devs)
ax.axis("off")

fig.patch.set_facecolor("white")
ax.set_facecolor("white")

for row_idx, row in summary.iterrows():
    y0 = n_devs - row_idx - 1

    ax.text(
        0.04,
        y0 + 0.78,
        row["developer"],
        ha="left",
        va="center",
        fontsize=18,
        fontweight="bold"
    )

    game_list = ", ".join(row["games"])

    ax.text(
        0.04,
        y0 + 0.67,
        game_list,
        ha="left",
        va="center",
        fontsize=8.8,
        color="#555555"
    )

    ax.text(
        0.04,
        y0 + 0.58,
        f"{row['frame_count']:,} analysed frames",
        ha="left",
        va="center",
        fontsize=8.8,
        color="#777777"
    )


    ax.text(
        0.30,
        y0 + 0.78,
        "Full palette",
        ha="left",
        va="center",
        fontsize=9,
        fontweight="bold"
    )

    draw_palette_bar(
        ax,
        x=0.40,
        y=y0 + 0.73,
        width=0.53,
        height=0.08,
        colors=row["full_palette"],
        weights=row["full_weights"]
    )

    ax.text(
        0.30,
        y0 + 0.62,
        "No-dark palette",
        ha="left",
        va="center",
        fontsize=9,
        fontweight="bold"
    )

    draw_palette_bar(
        ax,
        x=0.40,
        y=y0 + 0.57,
        width=0.53,
        height=0.08,
        colors=row["nodark_palette"],
        weights=row["nodark_weights"]
    )

    avg_bg = tuple(row["avg_color_rgb"] / 255.0)

    rect = patches.Rectangle(
        (0.04, y0 + 0.18),
        0.16,
        0.22,
        facecolor=avg_bg,
        edgecolor="#dddddd",
        linewidth=1.0
    )
    ax.add_patch(rect)

    ax.text(
        0.12,
        y0 + 0.31,
        row["avg_color_hex"],
        ha="center",
        va="center",
        fontsize=9.5,
        fontweight="bold",
        color=text_color_for_bg(avg_bg)
    )

    ax.text(
        0.12,
        y0 + 0.22,
        "Average color",
        ha="center",
        va="center",
        fontsize=7.8,
        color=text_color_for_bg(avg_bg)
    )


    start_x = 0.25
    cell_w = 0.105
    gap = 0.012

    for i, (metric, label) in enumerate(metrics):
        draw_metric_cell(
            ax,
            x=start_x + i * (cell_w + gap),
            y=y0 + 0.18,
            width=cell_w,
            height=0.22,
            label=label,
            value=row[metric],
            metric=metric
        )


ax.text(
    0.02,
    n_devs + 0.02,
    "Developer Comparison",
    ha="left",
    va="bottom",
    fontsize=22,
    fontweight="bold"
)

plt.tight_layout()
plt.savefig(output_path, dpi=220, bbox_inches="tight")
plt.close()

print(f"Saved: {output_path}")