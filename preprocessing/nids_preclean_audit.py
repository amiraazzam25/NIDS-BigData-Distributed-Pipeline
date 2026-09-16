from pyspark.sql import SparkSession
from pyspark.sql.functions import col, isnan, sum as spark_sum

INPUT_PATH = "s3a://nids-processed/nids-selected-35"

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

spark = (
    SparkSession.builder
    .appName("NIDS-PreClean-Audit")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

df = spark.read.parquet(INPUT_PATH)

rows = df.count()

print("\n" + "=" * 75)
print("NIDS PRE-CLEAN AUDIT")
print("=" * 75)
print(f"Rows    : {rows:,}")
print(f"Columns : {len(df.columns)}")

null_counts = df.select([
    spark_sum(col(c).isNull().cast("long")).alias(c)
    for c in df.columns
]).first().asDict()

nan_counts = df.select([
    spark_sum(isnan(col(c)).cast("long")).alias(c)
    for c in NUMERIC_COLS
]).first().asDict()

inf_counts = df.select([
    spark_sum(
        (
            (col(c) == float("inf")) |
            (col(c) == float("-inf"))
        ).cast("long")
    ).alias(c)
    for c in NUMERIC_COLS
]).first().asDict()

print("\nColumns containing invalid values:")
for c in df.columns:
    nulls = null_counts.get(c, 0)
    nans = nan_counts.get(c, 0)
    infs = inf_counts.get(c, 0)

    if nulls or nans or infs:
        print(
            f"{c:25s} "
            f"NULL={nulls:,} "
            f"NaN={nans:,} "
            f"Inf={infs:,}"
        )

print("=" * 75)

spark.stop()
