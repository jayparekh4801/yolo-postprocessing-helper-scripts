import cv2
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

classes = {
  0: "drone",
  1: "drone_body",
  2: "drone_landing_gear",
  3: "drone_motor"
}

colors = {
    0: (255, 0, 0),
    1: (0, 255, 0),
    2: (0, 0, 255),
    3: (0, 255, 255),
}

def drawBoundingBox(txt_file, image_file, output_dir, bg_color=(0, 0, 0), text_size=0.7):
    img = cv2.imread(image_file)
    h, w = img.shape[:2]
    file_name = image_file.split('/')[-1]
    model = output_dir.split('/')[-1]
    predicted_classes = set()
    
    with open(txt_file, 'r') as f:
        data = f.read()
        annotations = data.split('\n')
        for annotation in annotations[:-1]:
            values = annotation.split(' ')
            coords = list(map(float, values[1:-1]))
            class_id = int(values[0])
            cls = classes[class_id]
            color = colors[class_id]
            predicted_classes.add(class_id)
            conf = float(values[-1])
            points = [(int(x * w), int(y * h)) for x, y in zip(coords[0::2], coords[1::2])]

            pts = np.array(points, np.int32).reshape((-1, 1, 2))
            cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)
            # cv2.rectangle(img, points[0], points[1], color=color, thickness=2)
            label = f"{conf:.2f}"
            cv2.putText(img, label, points[0], cv2.FONT_HERSHEY_TRIPLEX, 0.7, color, 1)

        max_label_length = max([len(classes[class_id]) for class_id in predicted_classes], default=0)
        legend_width = max(200, 40 + max_label_length * 15)  # Adjust width based on label length
        legend_height = len(predicted_classes) * 30 + 10

        img = cv2.copyMakeBorder(img, 
                                top=0, 
                                bottom=0, 
                                left=0, 
                                right=legend_width, 
                                borderType=cv2.BORDER_CONSTANT, 
                                value=[0, 0, 0])
        _, image_new_width = img.shape[:2]
        legend_x_start = image_new_width - legend_width - 10
        # legend_y_start = h - legend_height - 10
        legend_y_start = 0

        cv2.rectangle(img, (legend_x_start, legend_y_start), (w, h), bg_color, -1)

        for i, class_id in enumerate(predicted_classes):
            color = colors[class_id]
            class_name = classes[class_id]
            cv2.rectangle(img, (legend_x_start + 10, legend_y_start + i * 30 + 10), (legend_x_start + 30, legend_y_start + i * 30 + 30), color, -1)
            cv2.putText(img, class_name, (legend_x_start + 40, legend_y_start + i * 30 + 25), cv2.FONT_HERSHEY_SIMPLEX, text_size, color, 2, cv2.LINE_AA)
        
        cv2.imwrite(os.path.join(output_dir, file_name), img)


if __name__ == "__main__":    # Example usage

    image_dir = "/Users/jaykumarparekh/Documents/Research/drone_postprocessing/attack_noise_scale0.4/images"
    label_dir = "/Users/jaykumarparekh/Documents/Research/drone_postprocessing/attack_noise_scale0.4/NONXAI_model0_labels"
    output_dir = "/Users/jaykumarparekh/Documents/Research/drone_postprocessing/attack_noise_scale0.4/non_xai_images"
    for file in os.listdir(image_dir):
        if file.endswith('.png') or file.endswith('.jpg'):
            file_path = os.path.join(image_dir, file)
            label_path = os.path.join(label_dir, f"{file[:-4]}.txt")
            if os.path.exists(label_path):
                drawBoundingBox(
                    label_path,
                    file_path,
                    output_dir
                )