import pandas as pd

df = pd.read_csv("training_data.csv")

df = df.dropna()
df = df[df["description"] != "description"]
df = df.drop_duplicates()

df.to_csv("cleaned_training_data.csv", index=False)

print("Clean done")