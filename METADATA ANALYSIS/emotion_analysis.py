import pandas as pd
import numpy as np

csv_path = "../metadata/images.csv"

df = pd.read_csv(csv_path)

H = df["avg_hue"] / 179.0
S = df["avg_saturation"] / 255.0
B = df["avg_brightness"] / 255.0

W = (np.cos(2 * np.pi * H) + 1) / 2

# Valdez and Mehrabian's formula
P_raw = 0.69 * B + 0.22 * S
A_raw = -0.31 * B + 0.60 * S

# Scale pleasure/valence to [-1, 1]
P = 2 * (P_raw / 0.91) - 1

# Scale arousal to [-1, 1]
A_min = -0.31
A_max = 0.60
A = 2 * ((A_raw - A_min) / (A_max - A_min)) - 1

def classify_emotion(p, a):
    if p < 0 and a < 0:
        return "melancholic"
    elif p >= 0 and a < 0:
        return "calm"
    elif p >= 0 and a >= 0:
        return "energetic"
    else:
        return "tense"

df["warmth"] = W.round(4)
df["valence"] = P.round(4)
df["arousal"] = A.round(4)
df["emotion_label"] = [
    classify_emotion(p, a)
    for p, a in zip(P, A)
]

df.to_csv(csv_path, index=False)

print("Emotion analysis complete.")