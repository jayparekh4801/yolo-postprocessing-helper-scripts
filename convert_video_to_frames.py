import cv2
import os

# ====== CONFIG ======
video_path = "/Users/jaykumarparekh/Documents/Research/drone_postprocessing/drone_studio_videos/video9.MP4"     # Path to your video
output_folder = "/Users/jaykumarparekh/Documents/Research/drone_postprocessing/drone_studio_images"   # Folder to save frames
frame_prefix = "video9"             # Prefix for saved frames
# ====================

# Create output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Open video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Cannot open video.")
    exit()

frame_count = 0

save_every = 20  # Change this number

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % save_every == 0:
        resized_frame = cv2.resize(frame, (640, 640))
        frame_filename = os.path.join(
            output_folder, f"{frame_prefix}_{frame_count:05d}.jpg"
        )
        cv2.imwrite(frame_filename, resized_frame)
    frame_count += 1


cap.release()
print(f"Done! {frame_count} frames saved in '{output_folder}'")
