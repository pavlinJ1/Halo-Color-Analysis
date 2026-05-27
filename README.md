# Halo Color Analysis

This project analyzes the visual style of Halo cutscenes using frame-level color metadata.

The goal is to compare how different Halo games use brightness, saturation, contrast, dominant color palettes, and color-based emotional tone. The project extracts metadata from cutscene frames, maps the color features into a Pleasure–Arousal emotion space, and visualizes the results as palettes, timelines, coordinate systems, and comparison charts.

---

## Data extraction

Frames were extracted from cutscene videos downloaded from Youtube. A frame was extracted each second.

The main extraction script is:

```text
METADATA ANALYSIS/extraction.py
```
The extracted frames are stored in the `images/` directory.

The metadata is stored in:

- `metadata/images.csv`
- `metadata/games.csv`

`images.csv` contains one row per extracted frame. It includes frame-level color features such as brightness, saturation, contrast, hue, dominant colors, and emotion scores.

`games.csv` contains game-level information such as title, release year, developer, art director, and other contextual metadata.

The dataset contains 25991 cutscene frames.

For each frame, the script calculates:

- average hue
- average saturation
- average brightness
- brightness standard deviation
- saturation standard deviation
- contrast
- dark pixel percentage
- five dominant full-frame colors
- five dominant no-dark colors

### Full palette and no-dark palette

Two types of dominant palettes are created for each frame.

The full palette includes all pixels in the frame. This captures the overall cinematic darkness and visual weight of the image.
The no-dark palette excludes very dark pixels before extracting dominant colors. This helps reveal the visible color identity of the frame when black or near-black pixels dominate the image.

---

## Emotional model

The emotional analysis is implemented in:

```text
METADATA ANALYSIS/emotion_analysis.py
```

The project uses a two dimensional Pleasure–Arousal space based on Russell's circumplex model of affect. The axes are:

- **Pleasure**: pleasant vs. unpleasant visual tone
- **Arousal**: calm vs. intense visual tone

Color metadata is mapped into this space using the Valdez and Mehrabian's formula:

```text
P_raw = 0.69B + 0.22S
A_raw = -0.31B + 0.60S
```

Where:

- `B` is normalized brightness
- `S` is normalized saturation

The raw values are then normalized to a `-1` to `1` range.

The resulting coordinates are classified into four emotion labels:

| Pleasure | Arousal | Label |
|---|---|---|
| low | low | melancholic |
| high | low | calm |
| high | high | energetic |
| low | high | tense |

---

## Visualizations

All final visualizations are generated from scripts in the `VISUALISATION/` directory and saved to:

```text
VISUALISATION/output/
```
