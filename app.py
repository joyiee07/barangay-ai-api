from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

# Load trained model
model = joblib.load("incident_model.pkl")
num_classes = len(model.classes_)

print(f"AI Model loaded successfully! ({num_classes} classes)")

# Dynamic threshold: ~2x random chance
RANDOM_CHANCE = (1 / num_classes) * 100
THRESHOLD = max(RANDOM_CHANCE * 2.5, 15)  # At least 15%, or 2.5x random

print(f"Confidence threshold: {THRESHOLD:.1f}% (random chance: {RANDOM_CHANCE:.1f}%)")

@app.route("/classify", methods=["POST"])
def classify():
    try:
        data = request.get_json()

        if not data or "description" not in data:
            return jsonify({"error": "Missing 'description' field"}), 400

        description = str(data["description"]).strip()

        if description == "":
            return jsonify({"error": "Empty description"}), 400

        # Prediction
        prediction = model.predict([description])[0]
        probabilities = model.predict_proba([description])[0]
        confidence = float(max(probabilities) * 100)

        print("INPUT:", description)
        print("PRED:", prediction)
        print("PROB:", probabilities)
        print("CONF:", confidence)

        # Dynamic threshold based on number of classes
        if confidence < THRESHOLD:
             prediction = prediction

        # Also return top 3 predictions for debugging
        top3_idx = np.argsort(probabilities)[-3:][::-1]
        top3 = [
            {
                "label": str(model.classes_[i]),
                "confidence": round(float(probabilities[i]) * 100, 2)
            }
            for i in top3_idx
        ]

        return jsonify({
            "prediction": prediction,
            "confidence": round(confidence, 2),
            "top_predictions": top3
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)