from flask import Flask, request, jsonify
import cv2
import numpy as np
import time
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import torch
import time



model = None
processor = None

app = Flask(__name__)

def init_model():
    global model, processor

    # Load Processor & VLA
    print("Initialize model...")
    start = time.time()
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b",
        # attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
        torch_dtype=torch.float32,
        #low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    end = time.time()
    print("Initialize model time:", end-start)


    # Quantize model
    print("Quantize model...")
    start = time.time()
    quantized_vla = torch.quantization.quantize_dynamic(
        vla, {torch.nn.Linear}, dtype=torch.qint8
    )
    quantized_vla.to("cpu")
    end = time.time()
    print("Quantize model time:", end-start)

    
    model = quantized_vla



def inference(image:Image, prompt:str) -> list:
    # Predict Action (7-DoF; un-normalize for BridgeData V2)
    start = time.time()
    print("Inference...")

    inputs = processor(prompt, image).to("cpu")
    action = model.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
    end = time.time()

    print("Predict action time:", end-start)

    return action


@app.route("/", methods=["POST"])
def infer():
    data = request.json.get("data", None)
    prompt = request.json.get("prompt", None)
    if data is None or prompt is None:
        return jsonify({"error": "No input data provided"}), 400
    
    img = np.array(data, dtype=np.uint8)
    img = np.reshape(img, (224,224,3))
    img = img[:, :, ::-1]
    img = Image.fromarray(img)

    action = inference(image=img, prompt=prompt)

    return jsonify({"result": action})


if __name__ == "__main__":
    init_model()

    # Run on port 80, all interfaces
    app.run(host="0.0.0.0", port=80)
