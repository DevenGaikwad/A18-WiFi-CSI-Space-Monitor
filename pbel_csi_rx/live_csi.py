import argparse
import csv
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import serial

TX_MAC = "a4:f0:0f:69:1a:94"
LABELS = ("NO_PRESENCE", "PERSON_PRESENT", "PERSON_MOVING")
FEATURES = (
    "amp_mean", "amp_std", "amp_p10", "amp_p90",
    "subcarrier_profile_std", "temporal_spread",
    "change_mean", "change_p90", "rssi_mean", "rssi_std",
)
WINDOW_US = 2_000_000


def parse_csi(line):
    if not line.startswith("CSI_DATA,"):
        return None

    fields = line.split(",", 8)
    if len(fields) != 9:
        raise ValueError("Wrong number of CSI fields")

    _, timestamp, mac, rssi, channel, mode, length, invalid, hex_data = fields

    if (
        mac.lower() != TX_MAC
        or int(channel) != 6
        or int(length) != 128
        or int(invalid) not in (0, 1)
        or len(hex_data) != 256
    ):
        raise ValueError("Unexpected CSI metadata")

    signed = np.frombuffer(bytes.fromhex(hex_data), dtype=np.int8).astype(float)

    # Identical to training: discard the first four CSI bytes.
    amplitude = np.hypot(signed[4::2], signed[5::2])
    return int(timestamp), amplitude, float(rssi)


def make_features(frames):
    a = np.asarray([frame[0] for frame in frames])
    rssi = np.asarray([frame[1] for frame in frames])
    delta = np.abs(np.diff(a, axis=0))
    flat = a.ravel()

    return np.asarray([[
        flat.mean(),
        flat.std(),
        np.percentile(flat, 10),
        np.percentile(flat, 90),
        a.mean(axis=0).std(),
        a.std(axis=0).mean(),
        delta.mean(),
        np.percentile(delta, 90),
        rssi.mean(),
        rssi.std(),
    ]], dtype=float)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="COM4")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    bundle = joblib.load(root / "models" / "lab_b_rf_rounds1_13.joblib")

    if tuple(bundle["feature_order"]) != FEATURES or bundle["window_seconds"] != 2:
        raise ValueError("Live features do not match the saved model")

    model = bundle["model"]
    if model.n_features_in_ != len(FEATURES):
        raise ValueError("Model expects a different number of features")

    output_dir = root / "data" / "live"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "_predictions.csv"
    )

    valid = 0
    rejected = 0
    predictions = 0
    start_us = None
    end_us = None
    previous_us = None
    frames = []
    last_frame_at = time.monotonic()
    last_warning_at = 0.0

    print(f"Opening receiver on {args.port} at 115200 baud.")
    print("Press Ctrl+C to stop.")

    try:
        with serial.Serial(args.port, 115200, timeout=1) as board, \
                output_path.open("x", newline="", encoding="utf-8") as file:

            writer = csv.writer(file)
            writer.writerow([
                "prediction_utc",
                "rx_window_start_us",
                "frames",
                "prediction",
                "rf_score_no_presence",
                "rf_score_person_present",
                "rf_score_person_moving",
                "rssi_mean_dbm",
                "change_mean",
            ])

            # Opening COM4 may restart the receiver.
            time.sleep(3)
            board.reset_input_buffer()

            while True:
                line = board.readline().decode(
                    "ascii", errors="replace"
                ).strip()

                try:
                    item = parse_csi(line)
                except ValueError:
                    rejected += 1
                    continue

                if item is None:
                    now = time.monotonic()
                    if now - last_frame_at > 5 and now - last_warning_at > 5:
                        print("No valid CSI for 5 seconds. Check transmitter and COM4.")
                        last_warning_at = now
                    continue

                timestamp_us, amplitude, rssi = item
                valid += 1
                last_frame_at = time.monotonic()

                if previous_us is not None and timestamp_us < previous_us:
                    print("Receiver restarted; aligning new windows.")
                    start_us = None
                    end_us = None
                    frames = []

                previous_us = timestamp_us

                if start_us is None:
                    # Wait for the next full two-second window.
                    start_us = (timestamp_us // WINDOW_US + 1) * WINDOW_US
                    end_us = start_us + WINDOW_US

                if timestamp_us < start_us:
                    continue

                while timestamp_us >= end_us:
                    if len(frames) >= 10:
                        x = make_features(frames)
                        scores = dict(zip(
                            model.classes_, model.predict_proba(x)[0]
                        ))
                        label = model.predict(x)[0]
                        when = datetime.now(timezone.utc).isoformat(
                            timespec="seconds"
                        )

                        writer.writerow([
                            when,
                            start_us,
                            len(frames),
                            label,
                            *(round(float(scores[c]), 4) for c in LABELS),
                            round(float(x[0, 8]), 2),
                            round(float(x[0, 6]), 4),
                        ])
                        file.flush()
                        predictions += 1

                        print(
                            f"{when}  {label:<15} "
                            f"frames={len(frames):2d}  "
                            f"RF scores empty/present/moving="
                            f"{scores[LABELS[0]]:.2f}/"
                            f"{scores[LABELS[1]]:.2f}/"
                            f"{scores[LABELS[2]]:.2f}"
                        )
                    else:
                        print(
                            f"Skipped window: only {len(frames)} valid CSI frames"
                        )

                    frames = []
                    start_us = end_us
                    end_us += WINDOW_US

                frames.append((amplitude, rssi))

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        print(
            f"Valid frames={valid}, rejected CSI lines={rejected}, "
            f"predicted windows={predictions}"
        )
        print(f"Prediction CSV: {output_path}")


if __name__ == "__main__":
    main()