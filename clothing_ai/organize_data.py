import os
import json
import shutil
from tqdm import tqdm

# ==== CONFIG ====
root_input = "/home/peter/uni/DeepFashion2/deepfashion2_original_images"  # contains train/, validation/, test/
output_root = "/home/peter/uni/clothing_ai/Data"

splits = ["train", "validation"]

# Make output structure
for split in splits:
    os.makedirs(os.path.join(output_root, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_root, split, "annotations"), exist_ok=True)


# ==== FUNCTION ====
def process_split(split_name):
    print(f"\n🔹 Processing {split_name} split...")

    anno_dir = os.path.join(root_input, split_name, "annos")
    img_dir = os.path.join(root_input, split_name, "image")
    out_img_dir = os.path.join(output_root, split_name, "images")
    out_anno_dir = os.path.join(output_root, split_name, "annotations")

    ann_files = [f for f in os.listdir(anno_dir) if f.endswith(".json")]
    count = 0

    for ann_file in tqdm(ann_files):
        with open(os.path.join(anno_dir, ann_file), "r") as f:
            data = json.load(f)

        # Each file can contain item1, item2, etc.
        for key, item in data.items():
            if not key.startswith("item"):
                continue

            if item.get("category_name", "").lower() != "short sleeve top":
                continue

            # Extract landmarks
            lms = item.get("landmarks", [])
            # Group into (x, y, visibility)
            landmarks = [lms[i:i+3] for i in range(0, len(lms), 3)]

            # Prepare new simplified annotation
            new_anno = {
                "image": ann_file.replace(".json", ".jpg"),
                "category_name": "short sleeve top",
                "landmarks": landmarks,
                "num_landmarks": len(landmarks)
            }   

            # Copy corresponding image
            img_name = ann_file.replace(".json", ".jpg")
            src_img_path = os.path.join(img_dir, img_name)
            dst_img_path = os.path.join(out_img_dir, img_name)

            if not os.path.exists(src_img_path):
                print(f"⚠️ Missing image for {ann_file}")
                continue

            shutil.copy(src_img_path, dst_img_path)

            # Save simplified JSON
            out_json_path = os.path.join(out_anno_dir, ann_file)
            with open(out_json_path, "w") as f:
                json.dump(new_anno, f)

            count += 1

    print(f"✅ Done {split_name}: {count} short-sleeve tops saved.")


# ==== RUN ====
for split in splits:
    process_split(split)

print("\n🎉 All splits processed and saved to:", output_root)
