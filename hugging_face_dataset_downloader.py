from huggingface_hub import snapshot_download
import zipfile
from pathlib import Path

# --- Configuration ---
REPO_ID = "lgrzybowski/seraphim-drone-detection-dataset"
LOCAL_DIR = Path("/Users/jaykumarparekh/Documents/Research/drone_postprocessing/hugging_face_drone_dataset") # TODO: change to your local directory

# --- Step 1: Download the entire repo ---
repo_path = Path(snapshot_download(repo_id=REPO_ID, repo_type="dataset", local_dir=LOCAL_DIR))

# --- Step 2: Unzip all .zip files in place ---
zip_files = list(repo_path.rglob("*.zip"))
print(f"Found {len(zip_files)} zip files to extract")

for zip_path in zip_files:
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(zip_path.parent)
        print(f"✅ Extracted: {zip_path.relative_to(repo_path)}")
        zip_path.unlink()  # remove the zip file
    except zipfile.BadZipFile:
        print(f"⚠️ Skipping invalid zip: {zip_path}")

print("🎉 All zips extracted and removed.")
print(f"📂 Dataset ready at: {repo_path.resolve()}")
