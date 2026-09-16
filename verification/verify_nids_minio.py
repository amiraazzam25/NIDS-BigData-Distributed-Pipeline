from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum as spark_sum

PATH = "s3a://nids-processed/nids-selected-35"

spark = (
    SparkSession.builder
    .appName("Verify-NIDS-MinIO")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

df = spark.read.parquet(PATH)

print("\n" + "=" * 60)
print("NIDS MINIO VERIFICATION")
print("=" * 60)

row_count = df.count()
column_count = len(df.columns)

print(f"Rows    : {row_count}")
print(f"Columns : {column_count}")

print("\nColumn names:")
for i, name in enumerate(df.columns, start=1):
    print(f"{i:02d}. {name}")

print("\nSchema:")
df.printSchema()

print("\nSample:")
df.select(
    "Timestamp",
    "Dst Port",
    "Protocol" if "Protocol" in df.columns else "Fwd Seg Size Min",
    "Flow Duration",
    "Label"
).show(10, truncate=False)

print("\nNULL COUNTS:")
null_counts = df.select([
    spark_sum(col(c).isNull().cast("int")).alias(c)
    for c in df.columns
]).collect()[0].asDict()

for feature, count in null_counts.items():
    print(f"{feature:25s} : {count}")

print("=" * 60)

if column_count == 37:
    print("✅ Correct: 35 features + Timestamp + Label = 37 columns")
else:
    print(f"❌ Expected 37 columns, found {column_count}")

spark.stop()
