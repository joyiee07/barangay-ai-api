from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

model = joblib.load("incident_model.pkl")

print("AI Model loaded successfully!")

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "online",
        "message": "Barangay AI API is active",
        "model_loaded": True
    })

@app.route("/classify", methods=["POST"])
def classify():
    try:
        data = request.get_json()

        if not data or "description" not in data:
            return jsonify({"error": "Missing description"}), 400

        description = str(data["description"]).strip()

        if description == "":
            return jsonify({"error": "Empty description"}), 400

        prediction = model.predict([description])[0]

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba([description])[0]
            confidence = float(max(probabilities) * 100)

            top3_idx = np.argsort(probabilities)[-3:][::-1]
            top3 = [
                {
                    "label": model.classes_[i],
                    "confidence": round(float(probabilities[i]) * 100, 2)
                }
                for i in top3_idx
            ]
        else:
            confidence = 0.0
            top3 = []

        return jsonify({
            "prediction": prediction,
            "confidence": round(confidence, 2),
            "top_predictions": top3
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run()
