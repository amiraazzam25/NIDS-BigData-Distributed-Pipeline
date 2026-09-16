from pyspark.sql import SparkSession

spark = (
    SparkSession.builder
    .appName("NIDS-Final-MinIO-Count")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

df = spark.read.parquet(
    "s3a://nids-processed/nids-selected-35"
)

rows = df.count()
cols = len(df.columns)

print("\n" + "=" * 60)
print("FINAL MINIO VERIFICATION")
print("=" * 60)
print("Rows    :", rows)
print("Columns :", cols)
print("=" * 60)

spark.stop()
