from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

# Load trained model
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
            return jsonify({
                "error": "Missing description"
            }), 400

        description = data["description"].strip()

        if description == "":
            return jsonify({
                "error": "Description cannot be empty"
            }), 400

        # Predict incident category
        prediction = model.predict([description])[0]

        confidence = 0.0
        top_predictions = []

        # Get classifier from pipeline
        classifier = model.named_steps["classifier"]

        # LinearSVC confidence estimation
        if hasattr(classifier, "decision_function"):

            scores = model.decision_function([description])[0]

            # Convert decision scores to pseudo probabilities
            exp_scores = np.exp(scores - np.max(scores))
            probabilities = exp_scores / np.sum(exp_scores)

            confidence = float(np.max(probabilities) * 100)

            classes = classifier.classes_

            top3_idx = np.argsort(probabilities)[-3:][::-1]

            top_predictions = [
                {
                    "label": classes[i],
                    "confidence": round(float(probabilities[i]) * 100, 2)
                }
                for i in top3_idx
            ]

        print("\n==============================")
        print("INPUT:", description)
        print("PREDICTION:", prediction)
        print("CONFIDENCE:", round(confidence, 2))
        print("==============================")

        return jsonify({
            "prediction": prediction,
            "confidence": round(confidence, 2),
            "top_predictions": top_predictions
        })

    except Exception as e:
        print("ERROR:", str(e))

        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )