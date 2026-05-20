import os
import pandas as pd
import matplotlib.pyplot as plt


images_path = "../metadata/images.csv"
games_path = "../metadata/games.csv"
output_dir = "output/timeline_emotion"

os.makedirs(output_dir, exist_ok=True)

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

# Adjust this if the curves are too jagged or too smooth
rolling_window = 50

for _, game_row in games_ordered.iterrows():
    game_id = game_row["game_id"]
    title = game_row["title"]

    sub = df[df["game_id"] == game_id].copy()

    if sub.empty:
        continue

    sub = sub.sort_values("timestamp")

    sub["valence_smooth"] = (
        sub["valence"]
        .rolling(window=rolling_window, center=True, min_periods=1)
        .mean()
    )

    sub["arousal_smooth"] = (
        sub["arousal"]
        .rolling(window=rolling_window, center=True, min_periods=1)
        .mean()
    )

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        sub["timestamp"],
        sub["valence_smooth"],
        label="Valence",
        linewidth=2.4
    )

    ax.plot(
        sub["timestamp"],
        sub["arousal_smooth"],
        label="Arousal",
        linewidth=2.4
    )

    ax.axhline(0, linewidth=1, alpha=0.7)

    ax.set_title(f"Emotional Timeline: {title}", fontsize=15, pad=14)
    ax.set_xlabel("Timestamp")

    ax.set_xlim(0, sub["timestamp"].max())
    ax.set_ylim(-1, 1)

    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", frameon=True)


    plt.tight_layout()

    output_path = os.path.join(output_dir, f"{game_id}.png")
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")