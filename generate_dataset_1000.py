import random
import pandas as pd

print("Generating dataset...")

classes = {
    "Theft (Pagnanakaw)": [
        "Someone stole my phone",
        "May nagnakaw ng cellphone ko",
        "Gin kawat ang akon cp",
        "Ninakaw ang motor ko",
        "May kumuha ng bag ko"
    ],
    "Fire (Sunog)": [
        "The house is on fire",
        "May sunog sa bahay",
        "Nagadilaab ang balay",
        "May apoy sa tindahan",
        "Smoke coming from house"
    ],
    "Flood (Baha)": [
        "Flood water entered the house",
        "May baha sa barangay",
        "Lubog ang kalsada",
        "Ginsulod sang baha ang balay",
        "Umaapaw ang ilog"
    ],
    "Accident (Aksidente)": [
        "Two vehicles collided",
        "May aksidente sa kalsada",
        "Nagbanggaan ang motor",
        "Naaksidente ang rider",
        "Disgrasya sa daan"
    ],
    "Violence (Karahasan)": [
        "Two men are fighting",
        "May nag-aaway sa kalsada",
        "Binugbog ang lalaki",
        "May saksakan sa barangay",
        "May rambol sa plaza"
    ],
    "Robbery (Panghoholdap)": [
        "Store was robbed",
        "Hinoldap ang tindahan",
        "May holdaper sa jeep",
        "Tinutukan ng baril",
        "Kinuha ang pera"
    ],
    "Burglary (Akyat Bahay)": [
        "A thief entered the house",
        "May akyat bahay",
        "Pumasok ang magnanakaw",
        "Sinira ang pinto",
        "Nagsulod ang kawatan"
    ],
    "Noise Complaint (Reklamo sa Maingay)": [
        "Loud karaoke at night",
        "Maingay ang kapitbahay",
        "Hindi makatulog dahil sa ingay",
        "Sobrang lakas ng music",
        "May videoke buong gabi"
    ]
}

TARGET = 1000
labels = list(classes.keys())

rows = []

for i in range(TARGET):
    label = random.choice(labels)
    text = random.choice(classes[label])
    rows.append([text, label])

df = pd.DataFrame(rows, columns=["description", "label"])
df.to_csv("dataset_1000.csv", index=False)

print("DONE: dataset_1000.csv created")
print("Rows:", len(df))