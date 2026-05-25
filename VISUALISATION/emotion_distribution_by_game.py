import os
import colorsys
import pandas as pd
import matplotlib.pyplot as plt


images_path = "../metadata/images.csv"
games_path = "../metadata/games.csv"
output_path = "output/emotion_distribution_by_game.png"

os.makedirs("output", exist_ok=True)

df = pd.read_csv(images_path)
games = pd.read_csv(games_path)

games["game_order"] = range(len(games))

df = df.merge(
    games[["game_id", "title", "game_order"]],
    on="game_id",
    how="left"
)

emotion_order = ["melancholic", "calm", "energetic", "tense"]

counts = (
    df.groupby(["title", "emotion_label"])
    .size()
    .unstack(fill_value=0)
)

counts = counts.reindex(columns=emotion_order, fill_value=0)

title_order = (
    games.sort_values("game_order")["title"]
    .tolist()
)

counts = counts.reindex(title_order)

percentages = counts.div(counts.sum(axis=1), axis=0) * 100


fixed_hue = 0.62

label_colors = {
    "melancholic": colorsys.hsv_to_rgb(fixed_hue, 0.25, 0.35),
    "calm": colorsys.hsv_to_rgb(fixed_hue, 0.25, 0.85),
    "energetic": colorsys.hsv_to_rgb(fixed_hue, 0.85, 0.85),
    "tense": colorsys.hsv_to_rgb(fixed_hue, 0.85, 0.35),
}

ax = percentages.plot(
    kind="bar",
    stacked=True,
    figsize=(13, 6),
    color=[label_colors[e] for e in emotion_order],
    edgecolor="white",
    linewidth=0.5
)

plt.ylabel("Percentage of frames")
plt.xlabel("Game")
plt.title("Emotion Label Distribution by Game")
plt.xticks(rotation=35, ha="right")

plt.legend(
    title="Emotion label",
    bbox_to_anchor=(1.05, 1),
    loc="upper left"
)

plt.tight_layout()
plt.savefig(output_path, dpi=200, bbox_inches="tight")
plt.close()

print(f"Saved: {output_path}")