import pandas as pd
import matplotlib.pyplot as plt

# =========================
# LOAD DATA
# =========================

df = pd.read_csv("../metadata/images.csv")

# =========================
# GROUP BY GAME
# =========================

game_stats = df.groupby("game_id").agg({
    "valence": "mean",
    "arousal": "mean"
}).reset_index()

print(game_stats)

# =========================
# PLOT EMOTIONAL SPACE
# =========================

plt.figure(figsize=(10, 8))

# axes
plt.axhline(0, color="gray", linewidth=1)
plt.axvline(0, color="gray", linewidth=1)

# quadrant labels
plt.text(-0.5, 0.5, "TENSE", fontsize=14, alpha=0.6, ha="center")
plt.text(0.5, 0.5, "ENERGETIC", fontsize=14, alpha=0.6, ha="center")
plt.text(-0.5, -0.5, "MELANCHOLIC", fontsize=14, alpha=0.6, ha="center")
plt.text(0.5, -0.5, "CALM", fontsize=14, alpha=0.6, ha="center")

# plot points
for _, row in game_stats.iterrows():
    x = row["valence"]
    y = row["arousal"]
    label = row["game_id"]

    plt.scatter(x, y, s=200)
    plt.text(x + 0.01, y + 0.01, label, fontsize=12)

# styling
plt.title("Halo Games — Emotional Space (Valence vs Arousal)")
plt.xlabel("Valence (Unpleasant → Pleasant)")
plt.ylabel("Arousal (Calm → Energetic)")

plt.xlim(-1, 1)
plt.ylim(-1, 1)

plt.grid(True, linestyle="--", alpha=0.3)

plt.savefig("output/emotion_space_games.png", dpi=300)
plt.show()

print("Saved: emotion_space_games.png")