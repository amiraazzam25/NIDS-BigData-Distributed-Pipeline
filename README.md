# NIDS Big Data Distributed Pipeline

An end-to-end Big Data pipeline for Network Intrusion Detection System (NIDS) traffic using Apache Kafka, Apache Spark, Distributed MinIO, XGBoost, Spark SQL, and Power BI.

## Project Objective

The project demonstrates how large-scale network traffic data can be ingested, processed, stored, analyzed, and used for machine learning in a distributed environment.

## 👥 Team Members

- Amira Azzam
- Ahmed Samy
- Randa Ashraf

## 👨‍🏫 Supervised by
Eng. Ahmed Hassan

## Pipeline Architecture

```text
NIDS Dataset
     |
     v
Python Producer
     |
     v
Apache Kafka
     |
     v
Spark Structured Streaming
     |
     v
Data Processing & Feature Selection
     |
     v
Distributed MinIO Storage
     |
     +-------------------+
     |                   |
     v                   v
Spark SQL              XGBoost
     |
     v
Spark Thrift Server
     |
     v
Power BI
```

## Three-Node Cluster

The project runs on three Ubuntu virtual machines.

### Node 1
- Python Producer
- Apache Kafka
- Spark Master
- MinIO

### Node 2
- Spark Worker
- MinIO

### Node 3
- Spark Worker
- MinIO

The three nodes communicate over the cluster network.

## Technologies

- Python
- Apache Kafka
- Apache Spark
- Spark Structured Streaming
- Spark SQL
- Spark Thrift Server
- MinIO
- XGBoost
- Power BI

## Data Ingestion

A Python producer reads NIDS network traffic records and publishes them to an Apache Kafka topic.

Kafka acts as the ingestion and messaging layer between the data producer and Spark.

## Stream Processing

Spark Structured Streaming continuously consumes records from Kafka.

The streaming pipeline performs the required transformations and writes the processed data to MinIO object storage.

## Data Preprocessing

The preprocessing stage includes data auditing, missing-value handling, and preparation of the features used by the processing and machine learning pipelines.

The final feature set contains 35 selected features.

## Distributed Storage

MinIO is deployed across the three cluster nodes to provide distributed object storage for the processed NIDS data.

## Machine Learning

XGBoost is used as the final machine learning model for network intrusion classification.

The repository contains the cluster training scripts and the generated training summary.

## Analytics and Visualization

Spark SQL is used to query and aggregate the processed NIDS data.

Spark Thrift Server provides a SQL interface that allows the analytical results to be consumed by Power BI for visualization and dashboard creation.

## Repository Structure

```text
NIDS-BigData-Distributed-Pipeline/
|
|-- producer/
|   `-- nids_producer.py
|
|-- spark/
|   `-- nids_stream_minio.py
|
|-- preprocessing/
|   |-- build_medians.py
|   |-- nids_preclean_audit.py
|   |-- write_clean_nids.py
|   `-- nids_medians.json
|
|-- machine_learning/
|   |-- ml_train_xgb_cluster.py
|   |-- ml_train_xgb_resume.py
|   `-- xgb_training_summary_resume.json
|
|-- verification/
|   |-- final_minio_count.py
|   `-- verify_nids_minio.py
|
|-- .gitignore
`-- README.md
```

## Dataset

The complete NIDS dataset is not included in this repository because of its large size.

## Verification

The repository includes verification scripts for checking the data stored in MinIO and validating the final record counts.

## Big Data Components

The project covers the main stages of a Big Data pipeline:

- Data ingestion with Kafka
- Stream processing with Spark Structured Streaming
- Distributed computation using a Spark cluster
- Distributed object storage using MinIO
- Data preprocessing and feature selection
- Machine learning using XGBoost
- Analytical queries using Spark SQL
- Visualization using Power BI
