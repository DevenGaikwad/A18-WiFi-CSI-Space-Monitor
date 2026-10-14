from collections import Counter
from pathlib import Path
import csv

project = Path(__file__).resolve().parent
source = project / "data" / "raw"
destination = project / "data" / "trimmed"
destination.mkdir(parents=True, exist_ok=True)

cutoff_us = 20_000_000
totals = Counter()
files = sorted(source.glob("*.csv"))

if not files:
    raise SystemExit(f"No CSV files found in {source}")

for path in files:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames
        rows = list(reader)

    if not rows or columns is None:
        raise ValueError(f"Empty or invalid CSV: {path.name}")

    kept = []
    for row in rows:
        payload = bytes.fromhex(row["csi_hex"])
        if len(payload) != int(row["csi_len"]):
            raise ValueError(f"CSI length mismatch in {path.name}")
        if int(row["rx_timestamp_us"]) >= cutoff_us:
            kept.append(row)

    if not kept:
        raise ValueError(f"No rows remain after trimming: {path.name}")

    output = destination / path.name
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(kept)

    totals[kept[0]["label"]] += len(kept)
    print(f"{path.name}: {len(rows)} raw -> {len(kept)} trimmed")

print(f"\nFiles: {len(files)}")
print(f"Total trimmed rows: {sum(totals.values())}")
print(f"Rows by class: {dict(totals)}")
print(f"Output folder: {destination}")