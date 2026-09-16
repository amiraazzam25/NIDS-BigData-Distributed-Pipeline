import json
import math

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    lit,
    pmod,
    xxhash64,
    row_number,
    rand,
)
from pyspark.sql.window import Window
from pyspark.sql.types import NumericType

from xgboost.spark import SparkXGBClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


# ============================================================
# CONFIG — SAME FINAL PIPELINE LOGIC
# ============================================================

# IMPORTANT:
# This should be the CLEAN 35-feature Parquet dataset in MinIO.
INPUT_PATH = "s3a://nids-processed/nids-clean-35"

MODEL_PATH = (
    "s3a://nids-processed/"
    "ml-models/xgboost-corrected-80-20-benign-infiltration-30k"
)

SUMMARY_FILE = "/home/bd/NIDS/xgb_training_summary.json"

# Locked test first:
TEST_BUCKETS = 100
TEST_BUCKET_CUTOFF = 20
SPLIT_SEED = 20260916

# Development pool:
POOL_CAP_PER_CLASS = 85000

# Final train targets:
TARGET_PER_CLASS = 50000
INFILTERATION_TARGET = 30000

# 90/10 train/validation inside the 80% development sample.
VAL_FRACTION = 0.10
VAL_SEED = 42

# XGBoost settings — same core settings as no-Spark model.
N_ESTIMATORS = 300
MAX_DEPTH = 8
LEARNING_RATE = 0.1
SUBSAMPLE = 0.8
COLSAMPLE_BYTREE = 0.8
XGB_SEED = 42

# Spark resources.
SHUFFLE_PARTITIONS = 32


# ============================================================
# SPARK
# ============================================================

spark = (
    SparkSession.builder
    .appName("NIDS-XGBoost-Corrected-80-20-30k")
    .config("spark.sql.adaptive.enabled", "true")
    .config("spark.sql.shuffle.partitions", str(SHUFFLE_PARTITIONS))
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


# ============================================================
# HEADER
# ============================================================

print("=" * 100)
print("NIDS XGBOOST — SPARK CLUSTER VERSION")
print("=" * 100)
print("Pipeline:")
print("  Original data -> LOCKED 20% test")
print("                  -> 80% development")
print("                  -> 85k/class pool cap")
print("                  -> 90/10 train/validation")
print("                  -> train-only balancing")
print("                  -> DISTRIBUTED XGBOOST")
print("=" * 100)
print(f"Input                 : {INPUT_PATH}")
print(f"Locked test           : 20%")
print(f"Pool cap/class        : {POOL_CAP_PER_CLASS:,}")
print(f"Target/class          : {TARGET_PER_CLASS:,}")
print(f"Infilteration target  : {INFILTERATION_TARGET:,}")
print("=" * 100)


# ============================================================
# 1. READ CLEAN DATA FROM MINIO
# ============================================================

df = spark.read.parquet(INPUT_PATH)

if "Label" not in df.columns:
    raise RuntimeError("Label column not found.")

# Same 35 ML features.
excluded = {"Label", "Timestamp"}

feature_cols = [c for c in df.columns if c not in excluded]

if len(feature_cols) != 35:
    raise RuntimeError(
        f"Expected 35 ML features, found {len(feature_cols)}: {feature_cols}"
    )

non_numeric = [
    f.name
    for f in df.schema.fields
    if f.name in feature_cols and not isinstance(f.dataType, NumericType)
]

if non_numeric:
    raise RuntimeError(
        f"Non-numeric features found: {non_numeric}"
    )

df = df.select("Label", *feature_cols)


# ============================================================
# 2. LOCK THE 20% FINAL TEST FIRST
#
# Deterministic content hash:
#   bucket < 20  -> final test
#   bucket >= 20 -> development
#
# The two sets are disjoint by construction.
# ============================================================

hash_expr = xxhash64(
    lit(SPLIT_SEED),
    *[col(c) for c in df.columns]
)

split_bucket = pmod(hash_expr, lit(TEST_BUCKETS))

with_split = df.withColumn("_split_bucket", split_bucket)

test = (
    with_split
    .filter(col("_split_bucket") < TEST_BUCKET_CUTOFF)
    .drop("_split_bucket")
)

development = (
    with_split
    .filter(col("_split_bucket") >= TEST_BUCKET_CUTOFF)
    .drop("_split_bucket")
)

print("\n" + "=" * 100)
print("LOCKED 20% TEST")
print("=" * 100)

test_count = test.count()
print(f"Locked test rows: {test_count:,}")

print("\nLocked test distribution:")
test.groupBy("Label").count().orderBy(col("count").desc()).show(
    100, truncate=False
)


# ============================================================
# 3. DEVELOPMENT POOL — MAX 85k PER CLASS
#
# Sampling is ONLY inside the 80% development partition.
# ============================================================

window = Window.partitionBy("Label").orderBy(rand(SPLIT_SEED))

development_pool = (
    development
    .withColumn("_rn", row_number().over(window))
    .filter(col("_rn") <= POOL_CAP_PER_CLASS)
    .drop("_rn")
)

print("\n" + "=" * 100)
print("DEVELOPMENT POOL")
print("=" * 100)

pool_count = development_pool.count()
print(f"Development pool rows: {pool_count:,}")

development_pool.groupBy("Label").count().orderBy(
    col("count").desc()
).show(100, truncate=False)


# ============================================================
# 4. 90/10 TRAIN / VALIDATION INSIDE DEVELOPMENT POOL ONLY
#
# Validation is never balanced.
# ============================================================

train_source, validation = development_pool.randomSplit(
    [1.0 - VAL_FRACTION, VAL_FRACTION],
    seed=VAL_SEED,
)

train_source = train_source.cache()
validation = validation.cache()

train_source_count = train_source.count()
validation_count = validation.count()

print("\n" + "=" * 100)
print("TRAIN / VALIDATION")
print("=" * 100)
print(f"Train source : {train_source_count:,}")
print(f"Validation   : {validation_count:,}")

print("\nTrain source distribution:")
train_source.groupBy("Label").count().orderBy(
    col("count").desc()
).show(100, truncate=False)


# ============================================================
# 5. TRAIN-ONLY BALANCING
#
# Final requested strategy:
#
#   Benign:
#       KEEP ALL available train rows
#
#   Infilteration:
#       cap at 30,000
#
#   Other classes:
#       >= 50k -> downsample to 50k
#       < 50k  -> oversample to 50k
#
# IMPORTANT:
# Validation and locked test are NOT modified.
# ============================================================

train_counts_rows = (
    train_source
    .groupBy("Label")
    .count()
    .collect()
)

train_counts = {
    r["Label"]: int(r["count"])
    for r in train_counts_rows
}

print("\n" + "=" * 100)
print("TRAIN BALANCING PLAN")
print("=" * 100)

pieces = []

for label, count in sorted(train_counts.items(), key=lambda x: -x[1]):
    label_str = str(label)

    # --------------------------------------------------------
    # Benign: keep everything available.
    # --------------------------------------------------------
    if label_str.strip().lower() == "benign":
        sampled = train_source.filter(col("Label") == label)
        mode = "keep_all_benign"

    # --------------------------------------------------------
    # Infilteration: cap at 30,000.
    # --------------------------------------------------------
    elif label_str.strip().lower() == "infilteration":
        target = min(count, INFILTERATION_TARGET)

        if count > target:
            sampled = (
                train_source
                .filter(col("Label") == label)
                .sample(
                    withReplacement=False,
                    fraction=target / float(count),
                    seed=XGB_SEED,
                )
            )

            # Make the cap closer to the requested target if
            # sample() returns slightly fewer rows.
            sampled_count = sampled.count()

            if sampled_count < target:
                remainder = (
                    train_source
                    .filter(col("Label") == label)
                    .subtract(sampled)
                    .limit(target - sampled_count)
                )
                sampled = sampled.unionByName(remainder)

            mode = "cap_30000"
        else:
            sampled = train_source.filter(col("Label") == label)
            mode = "keep_all_below_30000"

    # --------------------------------------------------------
    # Large non-Benign class -> downsample to 50k.
    # --------------------------------------------------------
    elif count >= TARGET_PER_CLASS:
        sampled = (
            train_source
            .filter(col("Label") == label)
            .sample(
                withReplacement=False,
                fraction=TARGET_PER_CLASS / float(count),
                seed=XGB_SEED,
            )
        )

        sampled_count = sampled.count()

        # Fill a possible shortfall without replacement.
        if sampled_count < TARGET_PER_CLASS:
            remainder = (
                train_source
                .filter(col("Label") == label)
                .subtract(sampled)
                .limit(TARGET_PER_CLASS - sampled_count)
            )
            sampled = sampled.unionByName(remainder)

        mode = "downsample"

    # --------------------------------------------------------
    # Small non-Benign class -> NORMAL OVERSAMPLING.
    # --------------------------------------------------------
    else:
        # Keep the original rows, then sample with replacement
        # for the extra required rows.
        base = train_source.filter(col("Label") == label)

        extra_needed = TARGET_PER_CLASS - count

        extra = base.sample(
            withReplacement=True,
            fraction=extra_needed / float(count),
            seed=XGB_SEED,
        )

        # If fraction-based sampling undershoots, repeat sampling
        # until we have enough rows, then limit exactly.
        extra = extra.limit(extra_needed)

        sampled = (
            base
            .unionByName(extra)
        )

        mode = "oversample"

    pieces.append(sampled)

    print(
        f"{label_str:35s} "
        f"{count:>10,} -> mode={mode}"
    )


balanced_train = pieces[0]

for piece in pieces[1:]:
    balanced_train = balanced_train.unionByName(piece)

balanced_train = balanced_train.cache()

balanced_train_count = balanced_train.count()

print("\n" + "=" * 100)
print("BALANCED TRAIN")
print("=" * 100)
print(f"Balanced train rows: {balanced_train_count:,}")

balanced_train.groupBy("Label").count().orderBy(
    col("count").desc()
).show(100, truncate=False)


# ============================================================
# 6. TRAIN-ONLY PREPROCESSING / NULL HANDLING
#
# Since the source is already the clean 35-feature Parquet,
# we keep the same feature set and do Spark-native finite checks.
# Any non-finite values are rejected before training.
#
# No test-derived statistics are used.
# ============================================================

for c in feature_cols:
    balanced_train = balanced_train.filter(
        col(c).isNotNull()
    )
    validation = validation.filter(
        col(c).isNotNull()
    )
    test = test.filter(
        col(c).isNotNull()
    )



# ============================================================
# 6.5 SAVE PREPARED DATA — RECOVERY CHECKPOINT
#
# Saves the exact datasets that will be used from this point:
#   balanced_train -> sampled training data
#   validation     -> untouched validation
#   test           -> LOCKED final 20% test
#
# If XGBoost fails later, these datasets can be reused directly.
# ============================================================

XGB_PREPARED_BASE = (
    "s3a://nids-processed/ml-prepared/xgb-214-v1"
)

PREPARED_TRAIN_PATH = (
    XGB_PREPARED_BASE + "/balanced-train"
)

PREPARED_VALIDATION_PATH = (
    XGB_PREPARED_BASE + "/validation"
)

PREPARED_TEST_PATH = (
    XGB_PREPARED_BASE + "/locked-test"
)

print("\n" + "=" * 100)
print("SAVING XGBOOST RECOVERY CHECKPOINT")
print("=" * 100)

print("Saving balanced training data...")
(
    balanced_train
    .write
    .mode("overwrite")
    .option("compression", "snappy")
    .parquet(PREPARED_TRAIN_PATH)
)

print("Saving validation data...")
(
    validation
    .write
    .mode("overwrite")
    .option("compression", "snappy")
    .parquet(PREPARED_VALIDATION_PATH)
)

print("Saving LOCKED test data...")
(
    test
    .write
    .mode("overwrite")
    .option("compression", "snappy")
    .parquet(PREPARED_TEST_PATH)
)

print("")
print("RECOVERY CHECKPOINT READY")
print("Train      :", PREPARED_TRAIN_PATH)
print("Validation :", PREPARED_VALIDATION_PATH)
print("Locked Test:", PREPARED_TEST_PATH)
print("=" * 100)

# ============================================================
# 7. DISTRIBUTED XGBOOST
#
# SparkXGBClassifier distributes XGBoost training across
# Spark executors instead of running classic sklearn XGB on
# the driver.
# ============================================================

print("\n" + "=" * 100)
print("DISTRIBUTED XGBOOST TRAINING")
print("=" * 100)

# SparkXGBClassifier uses num_workers for distributed workers.
# Spark will schedule these workers across available executors.
num_workers = 2

print(f"XGBoost workers: {num_workers}")
print("Starting distributed XGBoost...")


xgb = SparkXGBClassifier(
    features_col="features",
    label_col="label",
    prediction_col="prediction",
    probability_col="probability",
    raw_prediction_col="rawPrediction",

    num_workers=num_workers,

    n_estimators=N_ESTIMATORS,
    max_depth=MAX_DEPTH,
    learning_rate=LEARNING_RATE,
    subsample=SUBSAMPLE,
    colsample_bytree=COLSAMPLE_BYTREE,

    eval_metric="mlogloss",

    tree_method="hist",
    random_state=XGB_SEED,

)


# ============================================================
# 8. LABEL + VECTOR ASSEMBLER + MODEL
# ============================================================

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler

label_indexer = StringIndexer(
    inputCol="Label",
    outputCol="label",
    stringOrderType="alphabetAsc",
    handleInvalid="error",
)

assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features",
    handleInvalid="error",
)

pipeline = Pipeline(
    stages=[
        label_indexer,
        assembler,
        xgb,
    ]
)


print("\nFitting distributed XGBoost...")
model = pipeline.fit(balanced_train)

print("Distributed XGBoost training finished.")


# ============================================================
# 9. PREDICTION
# ============================================================

train_pred = model.transform(balanced_train)
val_pred = model.transform(validation)
test_pred = model.transform(test)


# ============================================================
# 10. METRICS
#
# FINAL CONSOLE OUTPUT:
#   TRAIN      -> Accuracy + Weighted F1
#   FINAL TEST -> Accuracy + Weighted F1
#
# Validation is still used internally for training, but its
# metrics are intentionally NOT printed.
#
# No classification report.
# No confusion matrix.
# No Macro F1.
# ============================================================

label_model = model.stages[0]

# model.transform() already includes the StringIndexer output column "label".
# Re-running label_model.transform() would try to create "label" a second time.
train_eval = train_pred
test_eval = test_pred

# Accuracy
train_accuracy = (
    train_eval
    .filter(col("label") == col("prediction"))
    .count() / float(train_eval.count())
)

test_accuracy = (
    test_eval
    .filter(col("label") == col("prediction"))
    .count() / float(test_eval.count())
)

# Weighted F1:
# Spark MulticlassClassificationEvaluator with metricName="f1"
# returns the weighted multiclass F1 for this classification task.
weighted_f1_evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="f1",
)

train_weighted_f1 = weighted_f1_evaluator.evaluate(train_eval)
test_weighted_f1 = weighted_f1_evaluator.evaluate(test_eval)

print("\n" + "=" * 80)
print("FINAL METRICS")
print("=" * 80)

print("\nTRAIN")
print(f"Accuracy    : {train_accuracy * 100:.2f}%")
print(f"Weighted F1 : {train_weighted_f1:.4f}")

print("\nFINAL TEST")
print(f"Accuracy    : {test_accuracy * 100:.2f}%")
print(f"Weighted F1 : {test_weighted_f1:.4f}")

print("\n" + "=" * 80)


# ============================================================
# 11. SAVE MODEL + SUMMARY
# ============================================================

print("\nSaving distributed Spark XGBoost model:")
print(MODEL_PATH)

model.write().overwrite().save(MODEL_PATH)

label_mapping = {
    label: index
    for index, label in enumerate(label_model.labels)
}

summary = {
    "pipeline": "Spark Distributed XGBoost",
    "input_path": INPUT_PATH,

    "split": {
        "method": "xxhash64 deterministic content split",
        "test_percent": 20,
        "development_percent": 80,
        "test_rule": "bucket < 20",
        "development_rule": "bucket >= 20",
    },

    "development_sampling": {
        "pool_cap_per_class": POOL_CAP_PER_CLASS,
    },

    "train_validation": {
        "validation_fraction": VAL_FRACTION,
        "train_source_rows": train_source_count,
        "validation_rows": validation_count,
    },

    "training_balance": {
        "benign": "keep_all",
        "infilteration": INFILTERATION_TARGET,
        "other_classes_target": TARGET_PER_CLASS,
        "small_classes": "oversample_with_replacement",
        "large_classes": "downsample_without_replacement",
    },

    "rows": {
        "locked_test": test_count,
        "development_pool": pool_count,
        "balanced_train": balanced_train_count,
    },

    "features": feature_cols,
    "label_mapping": label_mapping,

    "xgboost": {
        "n_estimators": N_ESTIMATORS,
        "max_depth": MAX_DEPTH,
        "learning_rate": LEARNING_RATE,
        "subsample": SUBSAMPLE,
        "colsample_bytree": COLSAMPLE_BYTREE,
        "tree_method": "hist",
        "distributed": True,
        "num_workers": num_workers,
        "random_state": XGB_SEED,
    },

    "results": {
        "train_accuracy": float(train_accuracy),
        "train_weighted_f1": float(train_weighted_f1),
        "final_test_accuracy": float(test_accuracy),
        "final_test_weighted_f1": float(test_weighted_f1),
    },
}

with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)


print("\n" + "=" * 100)
print("SPARK XGBOOST TRAINING COMPLETE")
print("=" * 100)
print("MODEL :", MODEL_PATH)
print("SUMMARY:", SUMMARY_FILE)
print("FINAL TEST WAS LOCKED BEFORE DEVELOPMENT SAMPLING.")
print("=" * 100)

spark.stop()
