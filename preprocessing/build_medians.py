import json
import numpy as np
import pandas as pd

CSV_FILE = "/home/bd/NIDS/NIDS.csv"
OUTPUT_FILE = "/home/bd/NIDS/nids_top40_medians.json"

TOP_40_FEATURES = [
    "Fwd Seg Size Min",
    "Bwd Pkt Len Std",
    "Fwd Header Len",
    "PSH Flag Cnt",
    "Fwd IAT Mean",
    "Dst Port",
    "Pkt Len Std",
    "Pkt Len Var",
    "Fwd URG Flags",
    "Fwd IAT Tot",

    "Fwd IAT Max",
    "Init Fwd Win Byts",
    "Fwd PSH Flags",
    "ACK Flag Cnt",
    "Protocol",
    "Fwd IAT Min",
    "Subflow Fwd Byts",
    "ECE Flag Cnt",
    "Flow IAT Mean",
    "Flow Pkts/s",

    "Bwd Seg Size Avg",
    "Fwd Pkt Len Max",
    "Bwd Header Len",
    "TotLen Bwd Pkts",
    "Bwd Pkt Len Mean",
    "Flow Duration",
    "RST Flag Cnt",
    "TotLen Fwd Pkts",
    "Bwd Pkts/s",
    "Bwd IAT Max",

    "Init Bwd Win Byts",
    "Bwd Pkt Len Max",
    "Fwd IAT Std",
    "Flow Byts/s",
    "Pkt Len Mean",
    "Fwd Pkt Len Std",
    "SYN Flag Cnt",
    "Bwd IAT Min",
    "Pkt Len Max",
    "Fwd Pkt Len Mean"
]

print("Reading NIDS dataset...")

df = pd.read_csv(CSV_FILE)

missing = [c for c in TOP_40_FEATURES if c not in df.columns]

if missing:
    print("❌ Missing columns:")
    for c in missing:
        print(c)
    raise SystemExit(1)

print(f"Rows          : {len(df)}")
print(f"Total columns : {len(df.columns)}")
print(f"Top features  : {len(TOP_40_FEATURES)}")

X = df[TOP_40_FEATURES].copy()

# Convert all 40 features to numeric
X = X.apply(
    pd.to_numeric,
    errors="coerce"
)

# +Infinity / -Infinity -> NaN
X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

# TCP window sentinel -1 -> NaN
for c in [
    "Init Fwd Win Byts",
    "Init Bwd Win Byts"
]:
    X[c] = X[c].replace(-1, np.nan)

# Calculate medians
medians = X.median()

# Safety fallback
medians = medians.fillna(0)

# Convert to normal Python float values
medians_dict = {
    feature: float(value)
    for feature, value in medians.items()
}

with open(OUTPUT_FILE, "w") as f:
    json.dump(
        medians_dict,
        f,
        indent=4
    )

print("=" * 60)
print("✅ TOP 40 MEDIANS CREATED")
print("=" * 60)

for feature, value in medians_dict.items():
    print(f"{feature:25s} -> {value}")

print("=" * 60)
print(f"Saved as: {OUTPUT_FILE}")
print(f"Number of medians: {len(medians_dict)}")
