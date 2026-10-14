import csv
import io
import zipfile
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score
)
from sklearn.model_selection import LeaveOneGroupOut

ZIP_PATH = Path("data/LAB_B_rounds1_15.zip")
MODEL_PATH = Path("models/lab_b_rf_rounds1_13.joblib")

CLASSES = ("NO_PRESENCE", "PERSON_PRESENT", "PERSON_MOVING")
FEATURES = (
    "amp_mean", "amp_std", "amp_p10", "amp_p90",
    "subcarrier_profile_std", "temporal_spread",
    "change_mean", "change_p90", "rssi_mean", "rssi_std",
)

def new_model():
    return RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

X, y, rounds, session_ids = [], [], [], []
raw_count = 0

with zipfile.ZipFile(ZIP_PATH) as archive:
    if archive.testzip() is not None:
        raise ValueError("The ZIP failed its integrity check.")

    names = sorted(
        name for name in archive.namelist()
        if name.lower().endswith(".csv")
    )
    if len(names) != 45:
        raise ValueError(f"Expected 45 CSV files; found {len(names)}.")

    for index, name in enumerate(names):
        round_number = index // 3 + 1
        rows = list(csv.DictReader(
            io.StringIO(archive.read(name).decode("utf-8-sig"))
        ))
        if not rows:
            raise ValueError(f"Empty recording: {name}")

        raw_count += len(rows)
        label = rows[0]["label"]
        session_id = rows[0]["session_id"]

        if label not in CLASSES or any(
            row["label"] != label
            or row["session_id"] != session_id
            or row["tx_mac"].lower() != "a4:f0:0f:69:1a:94"
            or row["channel"] != "6"
            or int(row["csi_len"]) != 128
            or len(row["csi_hex"]) != 256
            for row in rows
        ):
            raise ValueError(f"Unexpected recording format: {name}")

        # Same 28 non-overlapping, 2-second windows used in the evaluation.
        windows = [[] for _ in range(28)]

        for row in rows:
            window = (
                int(row["rx_timestamp_us"]) - 20_000_000
            ) // 2_000_000

            if 0 <= window < 28:
                signed = np.frombuffer(
                    bytes.fromhex(row["csi_hex"]), dtype=np.int8
                ).astype(float)

                # Signed I/Q byte pairs; skip the first four CSI bytes.
                amplitude = np.hypot(signed[4::2], signed[5::2])
                windows[window].append(
                    (amplitude, float(row["rssi_dbm"]))
                )

        for window_number, frames in enumerate(windows):
            if len(frames) < 10:
                raise ValueError(
                    f"Too few frames: {name}, window {window_number}"
                )

            amplitudes = np.asarray([item[0] for item in frames])
            rssi = np.asarray([item[1] for item in frames])
            changes = np.abs(np.diff(amplitudes, axis=0))
            flat = amplitudes.ravel()

            X.append([
                flat.mean(),
                flat.std(),
                np.percentile(flat, 10),
                np.percentile(flat, 90),
                amplitudes.mean(axis=0).std(),
                amplitudes.std(axis=0).mean(),
                changes.mean(),
                np.percentile(changes, 90),
                rssi.mean(),
                rssi.std(),
            ])
            y.append(label)
            rounds.append(round_number)
            session_ids.append(session_id)

X = np.asarray(X)
y = np.asarray(y)
rounds = np.asarray(rounds)
session_ids = np.asarray(session_ids)

for round_number in range(1, 16):
    actual = Counter(y[rounds == round_number])
    expected = Counter({label: 28 for label in CLASSES})
    if actual != expected:
        raise ValueError(f"Round {round_number} is incomplete: {actual}")

train = rounds <= 13
test = rounds >= 14

X_train = X[train]
y_train = y[train]
training_rounds = rounds[train]

# Validation: each fold excludes one entire training round.
validation_predictions = np.empty(len(X_train), dtype=object)

for train_index, validation_index in LeaveOneGroupOut().split(
    X_train, y_train, training_rounds
):
    fold_model = new_model().fit(
        X_train[train_index], y_train[train_index]
    )
    validation_predictions[validation_index] = fold_model.predict(
        X_train[validation_index]
    )

# Fit the final model using all 13 designated training rounds.
model = new_model().fit(X_train, y_train)
test_predictions = model.predict(X[test])

def print_results(title, actual, predicted):
    print(f"\n{title}")
    print(f"Accuracy: {accuracy_score(actual, predicted):.4f}")
    print(
        "Macro F1:",
        f"{f1_score(actual, predicted, labels=CLASSES, average='macro'):.4f}"
    )
    print("Class order:", ", ".join(CLASSES))
    print("Confusion matrix; rows=actual, columns=predicted:")
    print(confusion_matrix(actual, predicted, labels=CLASSES))
    print(classification_report(
        actual, predicted, labels=CLASSES,
        zero_division=0, digits=3
    ))

print(f"CSV files: {len(names)}; CSI rows: {raw_count}")
print(f"Training windows: {train.sum()}; test windows: {test.sum()}")

print_results(
    "Leave-one-round-out training validation",
    y_train, validation_predictions
)
print(
    "Training-set fit (optimistic):",
    f"{accuracy_score(y_train, model.predict(X_train)):.4f}"
)
print_results(
    "Held-out Rounds 14-15",
    y[test], test_predictions
)

for round_number in (14, 15):
    mask = rounds[test] == round_number
    correct = np.sum(y[test][mask] == test_predictions[mask])
    print(f"Round {round_number}: {correct}/{mask.sum()} correct windows")

print("\nFull-minute session majority votes:")
for session_id in dict.fromkeys(session_ids[test]):
    mask = session_ids[test] == session_id
    majority = Counter(test_predictions[mask]).most_common(1)[0][0]
    print(
        f"Round {rounds[test][mask][0]}: "
        f"actual={y[test][mask][0]}, predicted={majority}"
    )

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(
    {
        "model": model,
        "feature_order": FEATURES,
        "classes": CLASSES,
        "train_rounds": "1-13",
        "window_seconds": 2,
    },
    MODEL_PATH,
)
print(f"\nSaved model: {MODEL_PATH.resolve()}")