from flask import Flask, request, jsonify
import cv2
import numpy as np
import time
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import torch


model = None
processor = None
device = "cuda" if torch.cuda.is_available() else "cpu"

app = Flask(__name__)

def init_model():
    global model, processor

    # Load Processor & VLA
    print("Initialize model on", device, "...")
    start = time.time()

    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        trust_remote_code=True
    ).to(device)

    end = time.time()
    print("Model loaded in", end - start, "seconds.")


def inference(image: Image, prompt: str) -> list:
    # Predict Action (7-DoF; un-normalize for BridgeData V2)
    start = time.time()
    print("Running inference on", device, "...")

    # Move inputs to GPU
    inputs = processor(prompt, image).to(device)

    with torch.no_grad():
        action = model.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)

    end = time.time()
    print("Predict action time:", end - start)

    return action


@app.route("/", methods=["POST"])
def infer():
    # Convert data from json
    img_data = request.json.get("data", None)
    prompt_raw = request.json.get("prompt", None)

    if img_data is None or prompt_raw is None:
        return jsonify({"error": "No input data provided"}), 400
    
    prompt = f"In: What action should the robot take to {prompt_raw.lower()}?\nOut:"
    
    # Convert image data to type Image
    img = np.array(img_data, dtype=np.uint8)
    if img.size != 224 * 224 * 3:
        return jsonify({"error": "Input image wrong size. Should be 150528 (224*224*3)."}), 400
    
    img = np.reshape(img, (224, 224, 3))
    img = img[:, :, ::-1]  # BGR → RGB
    img = Image.fromarray(img)

    action = inference(image=img, prompt=prompt)

    return jsonify({"result": action.tolist()})


if __name__ == "__main__":
    init_model()
    app.run(host="0.0.0.0", port=80)
