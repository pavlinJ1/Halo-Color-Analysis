import cv2
import os
import pandas as pd
from tqdm import tqdm

video_path = "../videos/halo_infinite.mp4"
output_dir = "../images"
csv_path = "../metadata/images.csv"

start_index = 19555
crop_enabled = False # some halo games have 130px border on top and bottom
crop_pixels = 130

fps_extract = 1


os.makedirs(output_dir, exist_ok=True)

video_name = os.path.splitext(os.path.basename(video_path))[0]
game_id = video_name

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise ValueError(f"Could not open video: {video_path}")

video_fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

frame_interval = int(video_fps / fps_extract)

estimated_images = total_frames // frame_interval

data = []

frame_count = 0
saved_count = 0



with tqdm(total=estimated_images, desc=f"Processing {video_name}") as pbar:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            timestamp_sec = frame_count / video_fps

            if crop_enabled:
                h, w, _ = frame.shape
                if crop_pixels * 2 < h:  # safety check
                    frame = frame[crop_pixels:h - crop_pixels, :]
                else:
                    print("Warning: crop too large for this video, skipping crop.")

            img_index = start_index + saved_count
            filename = f"img_{img_index:05d}.jpg"
            filepath = os.path.join(output_dir, filename)

            cv2.imwrite(filepath, frame)

            data.append({
                "filename": filename,
                "game_id": game_id,
                "timestamp": round(timestamp_sec, 2)
            })

            saved_count += 1
            pbar.update(1)

        frame_count += 1

cap.release()


df = pd.DataFrame(data)

if os.path.exists(csv_path):
    df.to_csv(csv_path, mode='a', header=False, index=False)
else:
    df.to_csv(csv_path, index=False)

print(f"\nDone. Saved {saved_count} images from {video_name}.")
print(f"Next image index: {start_index + saved_count}")