import json

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import NumericType

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

from xgboost.spark import SparkXGBClassifier


# ============================================================
# PATHS
# ============================================================

TRAIN_PATH = (
    "s3a://nids-processed/"
    "ml-prepared/xgb-214-v1/balanced-train"
)

VALIDATION_PATH = (
    "s3a://nids-processed/"
    "ml-prepared/xgb-214-v1/validation"
)

TEST_PATH = (
    "s3a://nids-processed/"
    "ml-prepared/xgb-214-v1/locked-test"
)

MODEL_PATH = (
    "s3a://nids-processed/"
    "ml-models/xgboost-corrected-80-20-benign-infiltration-30k"
)

SUMMARY_FILE = "/home/bd/NIDS/xgb_training_summary_resume.json"


# ============================================================
# SPARK
# ============================================================

spark = (
    SparkSession.builder
    .appName("NIDS-XGBoost-RESUME")
    .config("spark.sql.adaptive.enabled", "true")
    .config("spark.sql.shuffle.partitions", "32")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


print("=" * 100)
print("NIDS XGBOOST — RESUME FROM RECOVERY CHECKPOINT")
print("=" * 100)


# ============================================================
# READ SAVED DATA
# ============================================================

balanced_train = spark.read.parquet(TRAIN_PATH)
validation = spark.read.parquet(VALIDATION_PATH)
test = spark.read.parquet(TEST_PATH)

print("Recovery checkpoint loaded successfully.")

print("Train path      :", TRAIN_PATH)
print("Validation path :", VALIDATION_PATH)
print("Locked test path:", TEST_PATH)


# ============================================================
# FEATURES
# ============================================================

excluded = {"Label", "Timestamp"}

feature_cols = [
    c for c in balanced_train.columns
    if c not in excluded
]

if len(feature_cols) != 35:
    raise RuntimeError(
        f"Expected 35 ML features, found {len(feature_cols)}"
    )

non_numeric = [
    f.name
    for f in balanced_train.schema.fields
    if f.name in feature_cols
    and not isinstance(f.dataType, NumericType)
]

if non_numeric:
    raise RuntimeError(
        f"Non-numeric features found: {non_numeric}"
    )

print(f"Features: {len(feature_cols)}")


# ============================================================
# XGBOOST
# ============================================================

num_workers = 2

xgb = SparkXGBClassifier(
    features_col="features",
    label_col="label",
    prediction_col="prediction",
    probability_col="probability",
    raw_prediction_col="rawPrediction",

    num_workers=num_workers,

    n_estimators=300,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,

    eval_metric="mlogloss",
    tree_method="hist",
    random_state=42,
)


# ============================================================
# PIPELINE
# ============================================================

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


print("\n" + "=" * 100)
print("STARTING DISTRIBUTED XGBOOST FROM SAVED CHECKPOINT")
print("=" * 100)
print("Workers: 2")
print("Max trees: 300")
print("No split.")
print("No resampling.")
print("Using saved balanced train + locked test.")
print("=" * 100)


# ============================================================
# TRAIN
# ============================================================

model = pipeline.fit(balanced_train)

print("\nDistributed XGBoost training finished.")


# ============================================================
# PREDICTIONS
# ============================================================

train_pred = model.transform(balanced_train)
val_pred = model.transform(validation)
test_pred = model.transform(test)

train_eval = train_pred
test_eval = test_pred


# ============================================================
# METRICS
# ============================================================

train_total = train_eval.count()
train_correct = (
    train_eval
    .filter(col("label") == col("prediction"))
    .count()
)

test_total = test_eval.count()
test_correct = (
    test_eval
    .filter(col("label") == col("prediction"))
    .count()
)

train_accuracy = train_correct / float(train_total)
test_accuracy = test_correct / float(test_total)

f1_evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="f1",
)

train_weighted_f1 = f1_evaluator.evaluate(train_eval)
test_weighted_f1 = f1_evaluator.evaluate(test_eval)


print("\n" + "=" * 100)
print("FINAL METRICS")
print("=" * 100)

print("\nTRAIN")
print(f"Rows        : {train_total:,}")
print(f"Accuracy    : {train_accuracy * 100:.2f}%")
print(f"Weighted F1 : {train_weighted_f1:.4f}")

print("\nFINAL LOCKED TEST")
print(f"Rows        : {test_total:,}")
print(f"Accuracy    : {test_accuracy * 100:.2f}%")
print(f"Weighted F1 : {test_weighted_f1:.4f}")


# ============================================================
# SAVE MODEL
# ============================================================

print("\nSaving model:")
print(MODEL_PATH)

model.write().overwrite().save(MODEL_PATH)

label_model = model.stages[0]

label_mapping = {
    label: index
    for index, label in enumerate(label_model.labels)
}

summary = {
    "pipeline": "Spark Distributed XGBoost Resume",
    "train_path": TRAIN_PATH,
    "validation_path": VALIDATION_PATH,
    "test_path": TEST_PATH,
    "features": feature_cols,
    "num_workers": num_workers,
    "label_mapping": label_mapping,
    "results": {
        "train_accuracy": float(train_accuracy),
        "train_weighted_f1": float(train_weighted_f1),
        "final_test_accuracy": float(test_accuracy),
        "final_test_weighted_f1": float(test_weighted_f1),
    },
}

with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
    json.dump(
        summary,
        f,
        indent=2,
        ensure_ascii=False,
    )

print("\n" + "=" * 100)
print("XGBOOST RESUME COMPLETE")
print("=" * 100)
print("MODEL  :", MODEL_PATH)
print("SUMMARY:", SUMMARY_FILE)
print("=" * 100)

spark.stop()
