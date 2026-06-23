import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

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

data = data[
    (data["description"] != "") &
    (data["label"] != "")
]

data = data.drop_duplicates()

# =========================
# SHOW CLASS COUNTS
# =========================
print("\n==============================")
print("CLASS COUNTS")
print("==============================")
print(data["label"].value_counts())

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
    test_size=0.20,
    random_state=42,
    stratify=y
)

# =========================
# LINEAR SVC MODEL
# =========================
model = Pipeline([
    (
        "vectorizer",
        TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=1,
            lowercase=True
        )
    ),
    (
        "classifier",
        LinearSVC(
            class_weight="balanced",
            random_state=42,
            max_iter=10000
        )
    )
])

# =========================
# TRAIN MODEL
# =========================
print("\nTraining LinearSVC Model...")
model.fit(X_train, y_train)

# =========================
# PREDICTIONS
# =========================
y_pred = model.predict(X_test)

# =========================
# ACCURACY
# =========================
accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")
print(f"Accuracy: {accuracy * 100:.2f}%")

# =========================
# CLASSIFICATION REPORT
# =========================
print("\n==============================")
print("CLASSIFICATION REPORT")
print("==============================")
print(classification_report(y_test, y_pred))

# =========================
# CONFUSION MATRIX
# =========================
print("\n==============================")
print("CONFUSION MATRIX")
print("==============================")

cm = confusion_matrix(y_test, y_pred)

cm_df = pd.DataFrame(
    cm,
    index=model.classes_,
    columns=model.classes_
)

print(cm_df)

# =========================
# SAVE MODEL
# =========================
joblib.dump(model, "incident_model.pkl")

print("\n==============================")
print("MODEL SAVED")
print("==============================")
print("Saved as incident_model.pkl")

# =========================
# SAMPLE TESTS
# =========================
test_cases = [
    "Nagbaha tungod sa kusog nga ulan",
    "May sunog sa barangay",
    "Ninakaw ang cellphone ko",
    "Gin kawat cp ko",
    "Gi kawat akong cp",
    "May nag-aaway sa kalsada",
    "Sobrang lakas ng karaoke",
    "May holdaper sa jeep",
    "May aksidente sa highway",
    "May umaakyat sa bakod",
    "May binugbog ng asawa",
    "Nasira ang waiting shed",
    "May nagsisigawan sa kalsada",
    "Na snatch akong bag",
    "Naholdap ko",
    "May rambol sa plaza"
]

print("\n==============================")
print("SAMPLE PREDICTIONS")
print("==============================")

for text in test_cases:

    prediction = model.predict([text])[0]

    print(f"\nINPUT: {text}")
    print(f"PREDICTION: {prediction}")