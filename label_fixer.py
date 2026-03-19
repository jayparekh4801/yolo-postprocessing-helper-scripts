import os

def fix_label_files(path):
    """
    Iterates through all .txt files in a folder, removes lines where the first value isn't '0',
    removes all commas, and overwrites the original file with the corrected content.
    """
    if not os.path.isdir(path):
        print(f"Error: The folder '{path}' was not found. Please check the path.")
        return

    print(f"Scanning folder: {path}\n")
    files_processed = 0

    for filename in os.listdir(path):
        if filename.endswith('.txt'):
            file_path = os.path.join(path, filename)
            
            try:
                with open(file_path, 'r') as file:
                    lines = file.readlines()

                # Filter lines: Keep only those starting with '0 '
                filtered_lines = [line for line in lines if line.strip().startswith('0 ')]

                # Join the lines and then replace commas
                modified_content = ''.join(filtered_lines).replace(',', '')

                with open(file_path, 'w') as file:
                    file.write(modified_content)
                
                print(f"✅ Fixed: {filename}")
                files_processed += 1

            except Exception as e:
                print(f"❌ Error processing {filename}: {e}")

    print(f"\nProcess complete. Total files fixed: {files_processed}.")


if __name__ == "__main__":
    folder_path = "/Users/jaykumarparekh/Documents/Research/drone_postprocessing/yolo_26_testing/test_data/ground_truth_labels_non_xai"
    fix_label_files(folder_path)