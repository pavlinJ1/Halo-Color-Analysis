import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


images_path = "../metadata/images.csv"
games_path = "../metadata/games.csv"
image_dir = "../images"
output_dir = "output/timeline_emotion"

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

rolling_window = 50


def add_frame_annotation(ax, row, y_col, label, xybox, label_color):
    frame_path = os.path.join(image_dir, row["filename"])

    if not os.path.exists(frame_path):
        print(f"Missing frame image: {frame_path}")
        return

    try:
        img = mpimg.imread(frame_path)
    except Exception as e:
        print(f"Could not read image: {frame_path} ({e})")
        return

    x = row["timestamp"]
    y = row[y_col]

    ax.scatter(
        x,
        y,
        s=40,
        color=label_color,
        zorder=6
    )

    imagebox = OffsetImage(
        img,
        zoom=0.09
    )

    ab = AnnotationBbox(
        imagebox,
        (x, y),
        xybox=xybox,
        xycoords="data",
        boxcoords="axes fraction",
        frameon=False,
        arrowprops=dict(
            arrowstyle="->",
            color="black",
            linewidth=1.2,
            shrinkA=4,
            shrinkB=4
        ),
        zorder=7
    )

    ax.add_artist(ab)

    ax.text(
        xybox[0],
        xybox[1] + 0.095,
        label,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=8,
        fontweight="bold",
        color=label_color,
        zorder=8
    )


for _, game_row in games_ordered.iterrows():
    game_id = game_row["game_id"]
    title = game_row["title"]

    sub = df[df["game_id"] == game_id].copy()

    if sub.empty:
        continue

    sub = sub.sort_values("timestamp").reset_index(drop=True)


    sub["pleasure_smooth"] = (
        sub["valence"]
        .rolling(window=rolling_window, center=True, min_periods=1)
        .mean()
    )

    sub["arousal_smooth"] = (
        sub["arousal"]
        .rolling(window=rolling_window, center=True, min_periods=1)
        .mean()
    )

    fig, ax = plt.subplots(figsize=(16, 7))

    ax.plot(
        sub["timestamp"],
        sub["pleasure_smooth"],
        label="Pleasure",
        linewidth=2.4
    )

    ax.plot(
        sub["timestamp"],
        sub["arousal_smooth"],
        label="Arousal",
        linewidth=2.4
    )

    ax.axhline(0, linewidth=1, alpha=0.7)

    ax.set_title(f"Emotion Timeline: {title}", fontsize=15, pad=14)
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Score")

    ax.set_xlim(0, sub["timestamp"].max())
    ax.set_ylim(-1, 1)

    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", frameon=True)


    plt.subplots_adjust(right=0.68)

    pleasure_min = sub.loc[sub["pleasure_smooth"].idxmin()]
    pleasure_max = sub.loc[sub["pleasure_smooth"].idxmax()]
    arousal_min = sub.loc[sub["arousal_smooth"].idxmin()]
    arousal_max = sub.loc[sub["arousal_smooth"].idxmax()]


    add_frame_annotation(
        ax,
        pleasure_min,
        "pleasure_smooth",
        "",
        xybox=(1.22, 0.84),
        label_color="#4c72b0"
    )

    add_frame_annotation(
        ax,
        pleasure_max,
        "pleasure_smooth",
        "",
        xybox=(1.22, 0.61),
        label_color="#4c72b0"
    )

    add_frame_annotation(
        ax,
        arousal_min,
        "arousal_smooth",
        "",
        xybox=(1.22, 0.38),
        label_color="#dd8452"
    )

    add_frame_annotation(
        ax,
        arousal_max,
        "arousal_smooth",
        "",
        xybox=(1.22, 0.15),
        label_color="#dd8452"
    )

    output_path = os.path.join(output_dir, f"{game_id}.png")
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")