from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

# Load model
model = joblib.load("incident_model.pkl")

print("AI Model loaded successfully!")

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Barangay AI API is running",
        "endpoints": {
            "classify": "POST /classify"
        }
    })

@app.route("/classify", methods=["POST"])
def classify():
    try:
        data = request.get_json()

        if not data or "description" not in data:
            return jsonify({"error": "Missing description"}), 400

        description = data["description"].strip()

        # Prediction
        prediction = model.predict([description])[0]

        # Check if model supports probabilities
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

        print("INPUT:", description)
        print("PREDICTION:", prediction)
        print("CONFIDENCE:", confidence)

        return jsonify({
            "prediction": prediction,
            "confidence": round(confidence, 2),
            "top_predictions": top3
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
