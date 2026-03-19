import cv2
import numpy as np
import os

classes = {
  0: "drone",
  1: "drone_body",
  2: "drone_landing_gear",
  3: "drone_motor"
}

colors = {
    0: (255, 255, 255),
    1: (255, 105, 180),
    2: (106, 90, 205),
    3: (0, 255, 255),
}

def process_video_with_labels(input_video_path, output_video_path, labels_data_dir, stride):

    # Open the input video file
    cap = cv2.VideoCapture(input_video_path)
    video_file_name = input_video_path.split('/')[-1].split('.')[0]
    label_files = os.listdir(labels_data_dir)
    bg_color=(0, 0, 0)
    text_size=0.7


    # Check if the video opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video file at {input_video_path}")
        return
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) # The new video will have a lower FPS

    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_index = 0
    max_label_length = max([len(class_name) for class_name in classes.values()], default=0)
    legend_width = max(200, 40 + max_label_length * 15)

    while frame_index < total_frames:
        # Set the video's current position to the desired frame index
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

        # Read the frame at the new position
        ret, img = cap.read()

        if not ret:
            break

        h, w = img.shape[:2]
        predicted_classes = set()
        label_file = f"{video_file_name}_{frame_index + 1}.txt"
        if label_file in label_files:
            label_file_path = os.path.join(labels_data_dir, label_file)
            print(label_file_path)
            with open(label_file_path, 'r') as f:
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
                    label = f"{conf:.2f}"
                    cv2.putText(img, label, points[0], cv2.FONT_HERSHEY_TRIPLEX, 0.5, color, 1)

                  # Adjust width based on label length
                # legend_height = len(predicted_classes) * 30 + 10

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

        print(f"processed {frame_index} frame.")
        cv2.imwrite(os.path.join(output_video_path, f"{video_file_name}_{frame_index + 1}.jpg"), img)
        # Advance the index by the stride
        frame_index += stride

    cap.release()
    cv2.destroyAllWindows()

# --- Example Usage ---

# 1. Define your video paths
if __name__ == "__main__":
    input_video = '/Users/jaykumarparekh/Documents/Research/drone_videos_for_inference/drone_moving.mov'
    postprocessed_frames = '/Users/jaykumarparekh/Documents/Research/drone_detection_synthetic_org/inference/videos/yolo11l-obb-xai-model-scale_07_10-drone_moving/postprocessed_frames'
    label_dir = '/Users/jaykumarparekh/Documents/Research/drone_detection_synthetic_org/inference/videos/yolo11l-obb-xai-model-scale_07_10-drone_moving/postprocessed_labels'
    process_video_with_labels(input_video, postprocessed_frames, label_dir, 1)