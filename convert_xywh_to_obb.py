import os

def convert_yolo_to_polygon(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(input_dir, filename), 'r') as f:
                lines = f.readlines()

            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                
                cls, x, y, w, h = map(float, parts)

                # Calculate corners
                x1, y1 = x - w/2, y - h/2
                x2, y2 = x + w/2, y - h/2
                x3, y3 = x + w/2, y + h/2
                x4, y4 = x - w/2, y + h/2

                # Format as: class_id x1 y1 x2 y2 x3 y3 x4 y4
                new_line = f"{int(cls)} {x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f} {x3:.6f} {y3:.6f} {x4:.6f} {y4:.6f}\n"
                new_lines.append(new_line)

            with open(os.path.join(output_dir, filename), 'w') as f:
                f.writelines(new_lines)

if __name__ == "__main__":
    input_folder = "/Users/jaykumarparekh/Documents/Research/drone_postprocessing/natural_images_test_dataset/labels"
    output_folder = "/Users/jaykumarparekh/Documents/Research/drone_postprocessing/natural_images_test_dataset/labels2"
    convert_yolo_to_polygon(input_folder, output_folder)