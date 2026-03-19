import cv2
import os

def combine_frames_to_video(image_folder, output_video_path, fps=24):
    img_paths = [
        os.path.join(image_folder, img) for img in os.listdir(image_folder)
        if img.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]

    img_paths.sort(key=os.path.getmtime)

    for file in img_paths:
        print(file)

    if not img_paths:
        raise ValueError("No image files found in the specified folder.")

    # === READ FIRST IMAGE TO GET FRAME SIZE ===
    frame = cv2.imread(img_paths[0])
    height, width, _ = frame.shape
    frame_size = (width, height)

    # === DEFINE VIDEO WRITER ===
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

    # === WRITE FRAMES TO VIDEO ===
    for img_path in img_paths:
        # img_path = os.path.join(image_folder, image_file)
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"Skipping unreadable file: {img_path}")
            continue
        resized_frame = cv2.resize(frame, frame_size)
        video_writer.write(resized_frame)

    video_writer.release()
    print(f"✅ Video created successfully at: {output_video_path}")

if __name__ == "__main__":
    image_folder = '/Users/jaykumarparekh/Documents/Research/drone_detection_synthetic_org/inference/videos/yolo11l-obb-xai-model-scale_07_10-drone_moving/postprocessed_frames'           # Folder containing images
    output_video_path = '/Users/jaykumarparekh/Documents/Research/drone_detection_synthetic_org/inference/videos/yolo11l-obb-xai-model-scale_07_10-drone_moving/postprocessed_drone_moving.mp4'      # Output video file path
    fps = 24
    combine_frames_to_video(image_folder, output_video_path, fps)