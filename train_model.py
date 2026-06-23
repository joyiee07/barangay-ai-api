import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# =========================
# LOAD DATASET
# =========================
data = pd.read_csv("dataset.csv")

# =========================
# CLEAN DATA
# =========================
data = data.dropna()

data["description"] = data["description"].astype(str).str.strip()
data["label"] = data["label"].astype(str).str.strip()

data = data[(data["description"] != "") & (data["label"] != "")]

data = data.drop_duplicates()

# =========================
# FEATURES & LABELS
# =========================
X = data["description"]
y = data["label"]

# =========================
# TRAIN / TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# MODEL PIPELINE
# =========================
model = Pipeline([
    ("vectorizer", TfidfVectorizer(
        ngram_range=(1, 3),
        lowercase=True,
        min_df=2,
        sublinear_tf=True
    )),
    ("classifier", LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        solver="lbfgs"
    ))
])

# =========================
# TRAIN MODEL
# =========================
model.fit(X_train, y_train)

# =========================
# PREDICTIONS
# =========================
y_pred = model.predict(X_test)

# =========================
# ACCURACY SCORE
# =========================
accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")
print(f"Accuracy: {accuracy * 100:.2f}%")

# =========================
# DETAILED REPORT
# =========================
print("\n==============================")
print("CLASSIFICATION REPORT")
print("==============================")
print(classification_report(y_test, y_pred))

# =========================
# SAVE MODEL
# =========================
joblib.dump(model, "incident_model.pkl")

print("\nModel saved as incident_model.pkl")

# =========================
# SAMPLE TESTS
# =========================
test_cases = [
    "Nagbaha tungod sa kusog nga ulan",
    "May sunog sa barangay",
    "Ninakaw ang cellphone ko",
    "Gin kawat cp ko",
    "May nag-aaway sa kalsada",
    "Sobrang lakas ng karaoke",
    "May holdaper sa jeep",
    "May aksidente sa highway"
]

print("\n==============================")
print("SAMPLE PREDICTIONS")
print("==============================")

for test in test_cases:
    pred = model.predict([test])[0]
    proba = model.predict_proba([test])[0]
    confidence = max(proba) * 100

    print(f"{test} → {pred} ({confidence:.1f}%)")