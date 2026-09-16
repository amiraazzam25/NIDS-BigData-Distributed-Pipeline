import csv
import json
import time
from confluent_kafka import Producer

CSV_FILE = "/home/bd/NIDS/NIDS.csv"

BOOTSTRAP_SERVER = "100.110.107.85:9092"
TOPIC = "nids-traffic"
PARTITIONS = 4

PROGRESS_EVERY = 100000

producer = Producer({
    "bootstrap.servers": BOOTSTRAP_SERVER,

    # Reliability
    "acks": "all",
    "enable.idempotence": True,

    # High throughput
    "compression.type": "lz4",
    "linger.ms": 20,
    "batch.num.messages": 10000,

    # Keep memory controlled
    "queue.buffering.max.messages": 200000,
    "queue.buffering.max.kbytes": 262144,

    # Reduce unnecessary delivery callback overhead
    "delivery.report.only.error": True,
})

sent = 0
start = time.time()

print("=" * 70)
print("NIDS HIGH-THROUGHPUT PRODUCER")
print("File       :", CSV_FILE)
print("Topic      :", TOPIC)
print("Partitions :", PARTITIONS)
print("=" * 70)

with open(
    CSV_FILE,
    "r",
    newline="",
    encoding="utf-8-sig",
    buffering=1024 * 1024
) as f:

    reader = csv.DictReader(f)

    for row in reader:

        payload = json.dumps(
            row,
            separators=(",", ":"),
            ensure_ascii=False
        ).encode("utf-8")

        partition = sent % PARTITIONS

        while True:
            try:
                producer.produce(
                    TOPIC,
                    value=payload,
                    partition=partition
                )
                break

            except BufferError:
                producer.poll(0.05)

        sent += 1

        if sent % 10000 == 0:
            producer.poll(0)

        if sent % PROGRESS_EVERY == 0:

            elapsed = time.time() - start
            rate = sent / elapsed if elapsed > 0 else 0

            print(
                f"Sent {sent:,} records "
                f"| {rate:,.0f} records/sec",
                flush=True
            )

print("Waiting for Kafka acknowledgements...")

remaining = producer.flush()

elapsed = time.time() - start

print("=" * 70)
print(f"Completed    : {sent:,} records")
print(f"Elapsed      : {elapsed / 60:.2f} minutes")
print(f"Average rate : {sent / elapsed:,.0f} records/sec")
print(f"Undelivered  : {remaining}")
print("=" * 70)
