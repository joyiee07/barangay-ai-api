from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)

# Load trained SVM model
model = joblib.load("incident_model.pkl")

# Get number of classes
num_classes = len(model.named_steps["classifier"].classes_)

print(f"AI Model loaded successfully! ({num_classes} classes)")

@app.route("/classify", methods=["POST"])
def classify():

    try:
        data = request.get_json()

        if not data or "description" not in data:
            return jsonify({
                "error": "Missing 'description' field"
            }), 400

        description = str(data["description"]).strip()

        if description == "":
            return jsonify({
                "error": "Empty description"
            }), 400

        # Predict using SVM
        prediction = model.predict([description])[0]

        print("INPUT:", description)
        print("PREDICTION:", prediction)

        # Fixed confidence for SVM
        confidence = 95.0

        return jsonify({
            "prediction": prediction,
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500

if __name__ == "__main__":
    app.run(debug=True)