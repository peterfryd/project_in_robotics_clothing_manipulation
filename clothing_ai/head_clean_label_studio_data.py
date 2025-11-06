import csv
import json
import os

# ==== CONFIG ====
INPUT_CSV = "./data/label_studio_csv_files/combined.csv"  # your CSV export
OUTPUT_DIR = "./data/annos"             # folder for per-image JSONs
CATEGORY_NAME = "short sleeve top"
NUM_LANDMARKS = 8  # adjust to match your dataset

os.makedirs(OUTPUT_DIR, exist_ok=True)

def parse_landmarks(kp_string, num_landmarks=NUM_LANDMARKS):
    """
    Converts Label Studio keypoint JSON string into [x, y, visibility] list
    """
    try:
        kp_data = json.loads(kp_string)
    except json.JSONDecodeError:
        print("⚠️ Failed to parse keypoints, returning zeros")
        return [[0, 0, 0]] * num_landmarks

    landmarks = [[0, 0, 0] for _ in range(num_landmarks)]
    for kp in kp_data:
        label_idx = int(kp["keypointlabels"][0]) - 1  # LabelStudio labels are 1-indexed
        x = round(kp["x"])
        y = round(kp["y"])
        visibility = 1 if x and y else 0
        if 0 <= label_idx < num_landmarks:
            landmarks[label_idx] = [x, y, visibility]

    return landmarks

# ==== PROCESS CSV ====
with open(INPUT_CSV, newline="", encoding="utf-8") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        full_image_name = os.path.basename(row["img"])
        # Keep only the part after the last '-'
        short_image_name = full_image_name.split("-")[-1]

        kp_string = row.get("kp-1", "[]")
        landmarks = parse_landmarks(kp_string, NUM_LANDMARKS)

        output_data = {
            "image": short_image_name,
            "category_name": CATEGORY_NAME,
            "landmarks": landmarks,
            "num_landmarks": NUM_LANDMARKS
        }

        # Save each image as a separate JSON
        output_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(short_image_name)[0]}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)

        print(f"✅ Saved: {output_path}")

print(f"🎉 Finished! JSONs saved in '{OUTPUT_DIR}'")
