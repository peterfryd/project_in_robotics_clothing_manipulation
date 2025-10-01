from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import cv2

app = Flask(__name__)


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
    img.save("image_im.jpg")

    return jsonify({"result": "hej"})


if __name__ == "__main__":
    # Run on port 80, all interfaces
    app.run(host="0.0.0.0", port=80)
