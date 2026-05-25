import os
import pandas as pd
import matplotlib.pyplot as plt

images_path = "../metadata/images.csv"
games_path = "../metadata/games.csv"
output_dir = "output/pleasure_arousal_by_game"

os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(images_path)
games = pd.read_csv(games_path)

games["game_order"] = range(len(games))

df = df.merge(
    games[["game_id", "title", "game_order"]],
    on="game_id",
    how="left"
)

games_ordered = games.sort_values("game_order")

for _, game_row in games_ordered.iterrows():
    game_id = game_row["game_id"]
    title = game_row["title"]

    sub = df[df["game_id"] == game_id]

    if sub.empty:
        continue

    plt.figure(figsize=(7, 6))

    plt.scatter(
        sub["valence"],
        sub["arousal"],
        s=6,
        alpha=0.25
    )

    plt.axhline(0, linewidth=1)
    plt.axvline(0, linewidth=1)

    plt.xlim(-1, 1)
    plt.ylim(-1, 1)

    plt.xlabel("Pleasure")
    plt.ylabel("Arousal")
    plt.title(f"Emotional Coordinate System: {title}")

    plt.text(-0.95, 0.88, "tense", fontsize=10)
    plt.text(0.45, 0.88, "energetic", fontsize=10)
    plt.text(0.55, -0.92, "calm", fontsize=10)
    plt.text(-0.95, -0.92, "melancholic", fontsize=10)

    plt.tight_layout()

    output_path = os.path.join(output_dir, f"{game_id}.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")