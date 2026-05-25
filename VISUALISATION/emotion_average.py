import os
import pandas as pd
import matplotlib.pyplot as plt


images_path = "../metadata/images.csv"
games_path = "../metadata/games.csv"
output_path = "output/emotion_average.png"

os.makedirs("output", exist_ok=True)

df = pd.read_csv(images_path)
games = pd.read_csv(games_path)

games["game_order"] = range(len(games))

df = df.merge(
    games[["game_id", "title", "game_order"]],
    on="game_id",
    how="left"
)

games_ordered = games.sort_values("game_order")

game_avg = (
    df.groupby(["game_id", "title", "game_order"])[["valence", "arousal"]]
    .mean()
    .reset_index()
    .sort_values("game_order")
)

overall_valence = df["valence"].mean()
overall_arousal = df["arousal"].mean()

fig, ax = plt.subplots(figsize=(11, 8))


ax.scatter(
    df["valence"],
    df["arousal"],
    s=8,
    alpha=0.06,
    color="gray"
)

color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

legend_handles = []
legend_labels = []


for i, (_, row) in enumerate(game_avg.iterrows()):
    x = row["valence"]
    y = row["arousal"]
    title = row["title"]

    color = color_cycle[i % len(color_cycle)]

    handle = ax.scatter(
        x,
        y,
        s=90,
        marker="x",
        linewidths=2.2,
        color=color,
        alpha=0.95,
        zorder=4
    )

    legend_handles.append(handle)
    legend_labels.append(title)

overall_handle = ax.scatter(
    overall_valence,
    overall_arousal,
    s=180,
    color="black",
    alpha=0.95,
    zorder=5
)

legend_handles.append(overall_handle)
legend_labels.append("All Halo Games")

ax.axhline(0, linewidth=1)
ax.axvline(0, linewidth=1)

ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)

ax.set_xlabel("Pleasure")
ax.set_ylabel("Arousal")
ax.set_title("Emotional Coordinate System with Game Averages", fontsize=16, pad=16)


ax.text(-0.95, 0.9, "tense", fontsize=10)
ax.text(0.62, 0.9, "energetic", fontsize=10)
ax.text(0.68, -0.93, "calm", fontsize=10)
ax.text(-0.95, -0.93, "melancholic", fontsize=10)

ax.grid(alpha=0.2)


ax.legend(
    legend_handles,
    legend_labels,
    title="Game",
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    frameon=True
)

plt.tight_layout()
plt.savefig(output_path, dpi=200, bbox_inches="tight")
plt.close()

print(f"Saved: {output_path}")