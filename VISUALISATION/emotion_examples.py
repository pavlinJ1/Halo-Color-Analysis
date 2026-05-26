import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


images_csv_path = "../metadata/images.csv"
image_dir = "../images"
output_path = "output/emotion_examples.png"

os.makedirs("output", exist_ok=True)

df = pd.read_csv(images_csv_path)

target_ids = ["img_00743", "img_07628", "img_10290", "img_17740"]


def find_row_by_stem(dataframe, stem):
    matches = dataframe[dataframe["filename"].astype(str).str.startswith(stem)]
    if matches.empty:
        return None
    return matches.iloc[0]


def classify_emotion(p, a):
    if p < 0 and a < 0:
        return "melancholic"
    elif p >= 0 and a < 0:
        return "calm"
    elif p >= 0 and a >= 0:
        return "energetic"
    else:
        return "tense"


example_rows = []

for stem in target_ids:
    row = find_row_by_stem(df, stem)

    if row is None:
        print(f"Could not find metadata row for: {stem}")
        continue

    filename = row["filename"]
    image_path = os.path.join(image_dir, filename)

    if not os.path.exists(image_path):
        print(f"Could not find image file: {image_path}")
        continue

    B_255 = float(row["avg_brightness"])
    S_255 = float(row["avg_saturation"])

    B = B_255 / 255.0
    S = S_255 / 255.0

    P_raw = 0.69 * B + 0.22 * S
    A_raw = -0.31 * B + 0.60 * S

    P = 2 * (P_raw / 0.91) - 1

    A = 2 * ((A_raw - (-0.31)) / (0.60 - (-0.31))) - 1

    emotion = classify_emotion(P, A)

    example_rows.append({
        "stem": stem,
        "filename": filename,
        "image_path": image_path,
        "B_255": B_255,
        "S_255": S_255,
        "B": B,
        "S": S,
        "P_raw": P_raw,
        "A_raw": A_raw,
        "P": P,
        "A": A,
        "emotion": emotion
    })


n = len(example_rows)

fig, axes = plt.subplots(
    nrows=n,
    ncols=3,
    figsize=(15, 4.2 * n),
    gridspec_kw={"width_ratios": [1.2, 1.1, 1.8]}
)

if n == 1:
    axes = np.array([axes])

for i, ex in enumerate(example_rows):
    ax_img = axes[i, 0]
    ax_pa = axes[i, 1]
    ax_txt = axes[i, 2]

    img = mpimg.imread(ex["image_path"])
    ax_img.imshow(img)
    ax_img.axis("off")


    ax_pa.axhline(0, linewidth=1, color="black", alpha=0.7)
    ax_pa.axvline(0, linewidth=1, color="black", alpha=0.7)

    ax_pa.scatter(
        ex["P"],
        ex["A"],
        s=120,
        zorder=3
    )

    ax_pa.set_xlim(-1, 1)
    ax_pa.set_ylim(-1, 1)
    ax_pa.set_aspect("equal", adjustable="box")

    ax_pa.set_xlabel("Pleasure")
    ax_pa.set_ylabel("Arousal")

    ax_pa.grid(alpha=0.2)

    ax_pa.text(-0.95, 0.88, "tense", fontsize=8)
    ax_pa.text(0.52, 0.88, "energetic", fontsize=8)
    ax_pa.text(0.62, -0.92, "calm", fontsize=8)
    ax_pa.text(-0.95, -0.92, "melancholic", fontsize=8)

    ax_pa.text(
        ex["P"] + 0.04,
        ex["A"] + 0.04,
        f"({ex['P']:.2f}, {ex['A']:.2f})",
        fontsize=8
    )

    ax_txt.axis("off")

    explanation = (
        "Original formula (Valdez & Mehrabian):\n"
        "P_raw = 0.69B + 0.22S\n"
        "A_raw = -0.31B + 0.60S\n\n"
        f"Frame values:\n"
        f"B = {ex['B_255']:.2f} / 255 = {ex['B']:.4f}\n"
        f"S = {ex['S_255']:.2f} / 255 = {ex['S']:.4f}\n\n"
        f"Calculation:\n"
        f"P_raw = 0.69({ex['B']:.4f}) + 0.22({ex['S']:.4f}) = {ex['P_raw']:.4f}\n"
        f"A_raw = -0.31({ex['B']:.4f}) + 0.60({ex['S']:.4f}) = {ex['A_raw']:.4f}\n\n"
        f"Normalized to [-1, 1]:\n"
        f"P = 2 * (P_raw / 0.91) - 1 = {ex['P']:.4f}\n"
        f"A = 2 * ((A_raw + 0.31) / 0.91) - 1 = {ex['A']:.4f}\n\n"
        f"Emotion label: {ex['emotion']}"
    )

    ax_txt.text(
        0.0,
        1.0,
        explanation,
        ha="left",
        va="top",
        fontsize=10,
        family="monospace"
    )

fig.suptitle(
    "Emotion Coordinate System Examples",
    fontsize=16,
    y=0.995
)


plt.tight_layout(rect=[0, 0.02, 1, 0.985])
plt.savefig(output_path, dpi=200, bbox_inches="tight")
plt.close()

print(f"Saved: {output_path}")