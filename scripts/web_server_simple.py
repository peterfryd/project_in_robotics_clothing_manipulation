from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route("/", methods=["POST"])
def infer():
    data = request.json.get("data", None)
    prompt = request.json.get("prompt", None)
    if data is None or prompt is None:
        return jsonify({"error": "No input data provided"}), 400
    
    return jsonify({"result": action})


if __name__ == "__main__":
    # Run on port 80, all interfaces
    app.run(host="0.0.0.0", port=80)
