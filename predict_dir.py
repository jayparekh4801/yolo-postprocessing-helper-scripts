from ultralytics import YOLO
import os
from . import xai_post_processing_1
from .plot_bounding_box import drawBoundingBox

MODE_PATH = "/Users/jaykumarparekh/Documents/Research/drone_postprocessing/models/yolo26l_xai_with_natural_images_model.pt"
model = YOLO(MODE_PATH)

DIR_PATH = "/Users/jaykumarparekh/Documents/Research/drone_postprocessing/natural_images_test_dataset/images"
project_name = "/Users/jaykumarparekh/Documents/Research/drone_postprocessing/yolo_model_testing"
subfolder_name = "xai_results"

model.predict(
    source=DIR_PATH,     # Input image path
    save=True,                          # Save output
    save_txt=True,                       # Set to True if you want text label output
    save_conf=True,                       # Save confidence scores on boxes
    conf = 0.01,
    project=project_name,
    name=subfolder_name,
    imgsz = 640,
)

inference_labels_path = os.path.join(project_name, subfolder_name, "labels")
inference_postprocessed_labels_path = os.path.join(project_name, subfolder_name, "postprocessed_labels")
os.makedirs(inference_postprocessed_labels_path, exist_ok=True)

config = {
        "association_dict" : {
            0: [1, 2, 3],
        }
    }
xai_post_processing_1.main(config, inference_labels_path, inference_postprocessed_labels_path)
output_postprocessed_images_dir = os.path.join(project_name, subfolder_name, "postprocessed_images")
os.makedirs(output_postprocessed_images_dir, exist_ok=True)
for file in os.listdir(DIR_PATH):
    if file.endswith('.png') or file.endswith('.jpg'):
        file_path = os.path.join(DIR_PATH, file)
        label_path = os.path.join(inference_postprocessed_labels_path, f"{file[:-4]}.txt")
        if os.path.exists(label_path):
            drawBoundingBox(
                label_path,
                file_path,
                output_postprocessed_images_dir
            )