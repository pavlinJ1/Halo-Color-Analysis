import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


images_path = "../metadata/images.csv"
games_path = "../metadata/games.csv"
output_path = "output/relative_averages.png"

os.makedirs("output", exist_ok=True)

df = pd.read_csv(images_path)
games = pd.read_csv(games_path)

# Preserve release order from games.csv
games["game_order"] = range(len(games))

df = df.merge(
    games[["game_id", "title", "game_order"]],
    on="game_id",
    how="left"
)

metrics = [
    ("avg_brightness", "Brightness", "0–255"),
    ("avg_saturation", "Saturation", "0–255"),
    ("contrast", "Contrast", "0–255"),
    ("warmth", "Warmth", "0–1"),
    ("valence", "Valence", "-1 to 1"),
    ("arousal", "Arousal", "-1 to 1")
]

summary = (
    df.groupby(["game_order", "title"])[[m[0] for m in metrics]]
    .mean()
    .reset_index()
    .sort_values("game_order")
)

titles = summary["title"].tolist()

# Earliest game at top, latest at bottom
y_positions = np.arange(len(titles))[::-1]

fig, axes = plt.subplots(
    nrows=2,
    ncols=3,
    figsize=(16, 8.5),
    sharey=False
)

axes = axes.flatten()

for ax, (metric, label, scale_label) in zip(axes, metrics):
    values = summary[metric].values

    # Lollipop stems
    ax.hlines(
        y=y_positions,
        xmin=values.min(),
        xmax=values,
        linewidth=2,
        alpha=0.65
    )

    # End points
    ax.scatter(
        values,
        y_positions,
        s=70,
        zorder=3
    )

    ax.set_title(f"{label}\n({scale_label})", fontsize=12)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(titles, fontsize=9)

    ax.grid(axis="x", alpha=0.25)

    if metric in ["valence", "arousal"]:
        ax.axvline(0, linewidth=1, alpha=0.7)

fig.suptitle(
    "Relative Average Color and Emotion Metrics",
    fontsize=17,
    y=0.98
)


plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(output_path, dpi=200, bbox_inches="tight")
plt.close()

print(f"Saved: {output_path}")