from flask import Flask, request, jsonify
import cv2

app = Flask(__name__)

@app.route("/", methods=["POST"])
def infer():
    data = request.json.get("data", None)
    if data is None:
        return jsonify({"error": "No input data provided"}), 400

    return jsonify({"result": data[0]})

if __name__ == "__main__":
    # Run on port 80, all interfaces
    app.run(host="0.0.0.0", port=80)
