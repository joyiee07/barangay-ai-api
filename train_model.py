import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

# Load dataset
data = pd.read_csv("dataset.csv")

X = data["description"]
y = data["label"]

# SVM Pipeline
model = Pipeline([
    ("vectorizer", TfidfVectorizer(
        ngram_range=(1, 2),
        lowercase=True,
        min_df=2,
        sublinear_tf=True
    )),
    ("classifier", LinearSVC())
])

# Train model
model.fit(X, y)

# Save model
joblib.dump(model, "incident_model.pkl")

print("Model trained successfully!")

# Get class labels
classes = model.named_steps["classifier"].classes_

print(f"Classes: {classes}")
print(f"Number of classes: {len(classes)}")

# Quick test
test_cases = [
    "Nagbaha tungod sa kusog nga ulan",
    "May sunog sa barangay",
    "Ninakaw ang cellphone ko",
    "Gin kawat cp ko"
]

for test in test_cases:
    pred = model.predict([test])[0]

    print(f"'{test}' → {pred}")
