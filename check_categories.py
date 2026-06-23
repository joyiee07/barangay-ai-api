import pandas as pd

# Load dataset
data = pd.read_csv("dataset.csv")

# Clean data (important)
data = data.dropna()
data["label"] = data["label"].astype(str).str.strip()

# Count categories
category_counts = data["label"].value_counts()

print("\n==============================")
print("DATASET CATEGORIES")
print("==============================\n")

for label, count in category_counts.items():
    print(f"{label} → {count} samples")

print("\n==============================")
print(f"TOTAL RECORDS: {len(data)}")
print(f"TOTAL CATEGORIES: {data['label'].nunique()}")
print("==============================")