from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, trim, when
from pyspark.sql.types import MapType, StringType

KAFKA_SERVER = "100.110.107.85:9092"
TOPIC = "nids-traffic"

OUTPUT_PATH = "s3a://nids-processed/nids-selected-35"
CHECKPOINT_PATH = "s3a://nids-checkpoints/nids-selected-35"

SELECTED_FEATURES = [
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

TCP_WINDOW_FEATURES = {
    "Init Fwd Win Byts",
    "Init Bwd Win Byts"
}

spark = (
    SparkSession.builder
    .appName("NIDS-FAST-Kafka-MinIO")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")
spark.conf.set("spark.sql.parquet.compression.codec", "snappy")

raw = (
    spark.readStream
    .format("kafka")
    .option("kafka.bootstrap.servers", KAFKA_SERVER)
    .option("subscribe", TOPIC)
    .option("startingOffsets", "earliest")
    .option("maxOffsetsPerTrigger", "250000")
    .load()
)

parsed = raw.select(
    from_json(
        col("value").cast("string"),
        MapType(StringType(), StringType())
    ).alias("data")
)

output_columns = [
    col("data")["Timestamp"].alias("Timestamp")
]

for feature in SELECTED_FEATURES:

    value = trim(col("data")[feature])

    numeric_value = (
        when(
            value.isNull()
            | value.isin(
                "",
                "NaN",
                "nan",
                "Infinity",
                "+Infinity",
                "-Infinity",
                "inf",
                "+inf",
                "-inf"
            ),
            None
        )
        .otherwise(value.cast("double"))
    )

    if feature in TCP_WINDOW_FEATURES:
        numeric_value = (
            when(numeric_value == -1, None)
            .otherwise(numeric_value)
        )

    output_columns.append(
        numeric_value.alias(feature)
    )

output_columns.append(
    col("data")["Label"].alias("Label")
)

selected = parsed.select(*output_columns)

assert len(selected.columns) == 37

print("=" * 70)
print("NIDS HIGH-PERFORMANCE STREAMING")
print("35 Features + Timestamp + Label = 37 columns")
print("Kafka partitions      : 4")
print("Max records per batch : 250000")
print("Output                :", OUTPUT_PATH)
print("Checkpoint            :", CHECKPOINT_PATH)
print("=" * 70)

query = (
    selected.writeStream
    .format("parquet")
    .outputMode("append")
    .option("path", OUTPUT_PATH)
    .option("checkpointLocation", CHECKPOINT_PATH)
    .start()
)

query.awaitTermination()
