import os
import pandas as pd
import matplotlib.pyplot as plt


images_path = "../metadata/images.csv"
games_path = "../metadata/games.csv"
output_dir = "output/timeline_color_features"

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

# Adjust this if the lines are too jagged or too smooth
rolling_window = 50

for _, game_row in games_ordered.iterrows():
    game_id = game_row["game_id"]
    title = game_row["title"]

    sub = df[df["game_id"] == game_id].copy()

    if sub.empty:
        continue

    sub = sub.sort_values("timestamp")

    sub["brightness_smooth"] = (
        sub["avg_brightness"]
        .rolling(window=rolling_window, center=True, min_periods=1)
        .mean()
    )

    sub["saturation_smooth"] = (
        sub["avg_saturation"]
        .rolling(window=rolling_window, center=True, min_periods=1)
        .mean()
    )

    sub["contrast_smooth"] = (
        sub["contrast"]
        .rolling(window=rolling_window, center=True, min_periods=1)
        .mean()
    )

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        sub["timestamp"],
        sub["brightness_smooth"],
        label="Brightness",
        linewidth=2.2
    )

    ax.plot(
        sub["timestamp"],
        sub["saturation_smooth"],
        label="Saturation",
        linewidth=2.2
    )

    ax.plot(
        sub["timestamp"],
        sub["contrast_smooth"],
        label="Contrast",
        linewidth=1.8,
        alpha=0.75
    )

    ax.set_title(f"Color Feature Timeline: {title}", fontsize=15, pad=14)
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Value, 0–255")

    ax.set_xlim(0, sub["timestamp"].max())
    ax.set_ylim(0, 255)

    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", frameon=True)


    plt.tight_layout()

    output_path = os.path.join(output_dir, f"{game_id}.png")
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")