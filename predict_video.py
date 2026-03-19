from ultralytics import YOLO
import os
from . import xai_post_processing_1
from . import generate_frames
from . import combine_frames
# Load your trained YOLOv8 model
MODE_PATH = "/Users/jaykumarparekh/Documents/Research/drone_postprocessing/models/yolo26l_xai_with_natural_images_model.pt"
model = YOLO(MODE_PATH)

VIDEO_PATH = "/Users/jaykumarparekh/Documents/Research/drone_postprocessing/drone_videos_for_inference/visible.mp4"

project_name = "/Users/jaykumarparekh/Documents/Research/drone_postprocessing/yolo_drone_video_testing"
Subfolder_name = "xai_vide_testing"

model.predict(
    source=VIDEO_PATH,     # Input video path
    save=True,                          # Save output
    save_txt=True,                       # Set to True if you want text label output
    save_conf=True,                       # Save confidence scores on boxes
    project=project_name,              # Output folder (created if not exists)
    name=Subfolder_name,                # Subfolder name
    vid_stride=3,                       # Process every frame; change if needed
    conf = 0.1,
    imgsz = 640,
)

inference_labels_path = os.path.join(project_name, Subfolder_name, "labels")
inference_postprocessed_labels_path = os.path.join(project_name, Subfolder_name, "postprocessed_labels")
os.makedirs(inference_postprocessed_labels_path, exist_ok=True)

config = {
        "association_dict" : {
            0: [1, 2, 3],
        }
    }
xai_post_processing_1.main(config, inference_labels_path, inference_postprocessed_labels_path)

inference_frames_dir = os.path.join(project_name, Subfolder_name, "inference_frames")
postprocessed_frames_dir = os.path.join(project_name, Subfolder_name, "postprocessed_frames")

os.makedirs(inference_frames_dir, exist_ok=True)
os.makedirs(postprocessed_frames_dir, exist_ok=True)

generate_frames.process_video_with_labels(
    VIDEO_PATH,
    postprocessed_frames_dir,
    inference_postprocessed_labels_path,
    stride=1
)

combine_frames.combine_frames_to_video(
    postprocessed_frames_dir,
    os.path.join(project_name, Subfolder_name, "postprocessed_" + os.path.basename(VIDEO_PATH).replace('.mp4', '.mp4')),
    24
)




