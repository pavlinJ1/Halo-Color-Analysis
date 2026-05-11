import pandas as pd
import numpy as np

csv_path = "metadata/images.csv"

df = pd.read_csv(csv_path)

# normalization
H = df["avg_hue"] / 179.0 # OpenCV hue is [0,179]
S = df["avg_saturation"] / 255.0
B = df["avg_brightness"] / 255.0

W = (np.cos(2 * np.pi * H) + 1) / 2
P_raw = 0.7 * B + 0.2 * S + 0.1 * W
A_raw = -0.3 * B + 0.6 * S + 0.1 * W

# scale [-1, 1]
A_raw = A_raw + 0.3
P = 2 * P_raw - 1
A = 2 * A_raw - 1

# classify emotion
def classify_emotion(p, a):
    if p < 0 and a < 0:
        return "melancholic"
    elif p >= 0 and a < 0:
        return "calm"
    elif p >= 0 and a >= 0:
        return "energetic"
    else:
        return "tense"

emotion_labels = [classify_emotion(p, a) for p, a in zip(P, A)]

df["warmth"] = W.round(4)
df["valence"] = P.round(4)
df["arousal"] = A.round(4)
df["emotion_label"] = emotion_labels

df.to_csv(csv_path, index=False)

print("Emotion analysis complete.")