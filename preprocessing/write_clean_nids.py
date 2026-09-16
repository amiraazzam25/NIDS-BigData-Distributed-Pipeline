import json

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, isnan, when

INPUT_PATH = "s3a://nids-processed/nids-selected-35"
OUTPUT_PATH = "s3a://nids-processed/nids-clean-35"
MEDIAN_FILE = "/home/bd/NIDS/nids_medians.json"

NUMERIC_COLS = [
    "Fwd Seg Size Min",
    "Bwd Header Len",
    "Pkt Len Max",
    "Init Fwd Win Byts",
    "Dst Port",
    "ECE Flag Cnt",
    "Bwd Pkt Len Std",
    "Fwd Header Len",
    "RST Flag Cnt",
    "Bwd Pkt Len Mean",
    "Tot Fwd Pkts",
    "PSH Flag Cnt",
    "Flow Byts/s",
    "Fwd Pkts/s",
    "Fwd IAT Min",
    "Fwd Act Data Pkts",
    "ACK Flag Cnt",
    "Flow Duration",
    "Bwd IAT Mean",
    "Init Bwd Win Byts",
    "Fwd IAT Max",
    "SYN Flag Cnt",
    "URG Flag Cnt",
    "Bwd Pkts/s",
    "Fwd IAT Tot",
    "Fwd PSH Flags",
    "Tot Bwd Pkts",
    "Flow IAT Max",
    "CWE Flag Count",
    "Fwd IAT Mean",
    "Fwd Pkt Len Max",
    "Bwd IAT Min",
    "Flow IAT Min",
    "Fwd IAT Std",
    "Idle Max"
]

TCP_WINDOW_COLS = {
    "Init Fwd Win Byts",
    "Init Bwd Win Byts"
}

spark = (
    SparkSession.builder
    .appName("NIDS-Final-Clean-Write")
    .config("spark.sql.adaptive.enabled", "true")
    .config("spark.sql.shuffle.partitions", "64")
    .config("spark.sql.parquet.compression.codec", "snappy")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

with open(MEDIAN_FILE, "r", encoding="utf-8") as f:
    medians = json.load(f)

print("=" * 75)
print("NIDS FINAL CLEANING")
print("=" * 75)

print("Medians:")
for c, v in medians.items():
    print(f"{c:25s}: {v}")

df = spark.read.parquet(INPUT_PATH)

# Defensive cleaning: NaN / +/-Infinity / TCP -1 -> NULL
clean = df

for c in NUMERIC_COLS:

    invalid = (
        col(c).isNull()
        | isnan(col(c))
        | (col(c) == float("inf"))
        | (col(c) == float("-inf"))
    )

    if c in TCP_WINDOW_COLS:
        invalid = invalid | (col(c) == -1)

    clean = clean.withColumn(
        c,
        when(invalid, None).otherwise(col(c))
    )

# Exact duplicates across all 37 columns
clean = clean.dropDuplicates()

# Fill NULL only in columns for which medians were calculated
clean = clean.na.fill({
    c: float(v)
    for c, v in medians.items()
})

assert len(clean.columns) == 37

print("\nWriting cleaned dataset...")
print("Output:", OUTPUT_PATH)

(
    clean.write
    .mode("overwrite")
    .option("compression", "snappy")
    .parquet(OUTPUT_PATH)
)

print("\n" + "=" * 75)
print("CLEAN DATASET WRITTEN SUCCESSFULLY")
print("=" * 75)
print("Expected rows   : 15,796,882")
print("Columns         : 37")
print("Output          :", OUTPUT_PATH)
print("=" * 75)

spark.stop()
