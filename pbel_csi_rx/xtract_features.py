import csv
import math
import statistics
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "data" / "trimmed"
OUTPUT = ROOT / "data" / "features" / "csi_windows_2s.csv"

LABELS = {"NO_PRESENCE", "PERSON_PRESENT", "PERSON_MOVING"}
START_US = 20_000_000
WINDOW_US = 2_000_000
WINDOWS_PER_SESSION = 28  # Complete windows from 20 to 76 seconds.


def percentile(values, fraction):
    ordered = sorted(values)
    location = (len(ordered) - 1) * fraction
    lower = int(location)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (
        ordered[upper] - ordered[lower]
    ) * (location - lower)


def rounded(value):
    return round(value, 6)


files = sorted(INPUT.glob("*.csv"))

if len(files) != 15:
    raise SystemExit(
        f"Expected the 15 pilot CSVs in {INPUT}; found {len(files)}."
    )

# The pilot contains three differently labeled recordings in each round.
for start in range(0, 15, 3):
    round_labels = {
        file.stem.split("Z_", 1)[-1]
        for file in files[start:start + 3]
    }
    if round_labels != LABELS:
        raise SystemExit(
            f"Round {start // 3 + 1} does not contain all three labels."
        )

features = []

for file_index, path in enumerate(files):
    windows = {i: [] for i in range(WINDOWS_PER_SESSION)}

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            time_us = int(row["rx_timestamp_us"])

            if not (
                START_US
                <= time_us
                < START_US + WINDOWS_PER_SESSION * WINDOW_US
            ):
                continue

            expected_label = path.stem.split("Z_", 1)[-1]
            if (
                row["session_id"] != path.stem
                or row["label"] != expected_label
            ):
                raise ValueError(
                    f"Unexpected label or session ID in {path.name}"
                )

            if (
                int(row["first_word_invalid"]) != 1
                or int(row["csi_len"]) != 128
            ):
                raise ValueError(
                    f"Unexpected CSI format in {path.name}"
                )

            raw = bytes.fromhex(row["csi_hex"])
            if len(raw) != 128:
                raise ValueError(
                    f"CSI byte count is not 128 in {path.name}"
                )

            signed = struct.unpack("128b", raw)

            # Each pair is imaginary, real. Skip the first four invalid bytes.
            amplitude = [
                math.hypot(signed[i], signed[i + 1])
                for i in range(4, 128, 2)
            ]

            window_number = (time_us - START_US) // WINDOW_US
            windows[window_number].append(
                (amplitude, int(row["rssi_dbm"]))
            )

    for window_number, frames in windows.items():
        if len(frames) < 10:
            raise ValueError(
                f"Only {len(frames)} frames in {path.name}, "
                f"window {window_number}"
            )

        amplitude_rows = [frame[0] for frame in frames]
        rssi = [frame[1] for frame in frames]

        all_amplitudes = [
            value
            for frame in amplitude_rows
            for value in frame
        ]

        mean_by_position = [
            statistics.fmean(frame[i] for frame in amplitude_rows)
            for i in range(62)
        ]

        std_by_position = [
            statistics.pstdev(frame[i] for frame in amplitude_rows)
            for i in range(62)
        ]

        consecutive_changes = [
            abs(current[i] - previous[i])
            for previous, current in zip(
                amplitude_rows, amplitude_rows[1:]
            )
            for i in range(62)
        ]

        features.append({
            "session_id": path.stem,
            "round_id": file_index // 3 + 1,
            "label": path.stem.split("Z_", 1)[-1],
            "window_start_s": 20 + 2 * window_number,
            "frame_count": len(frames),
            "amp_mean": rounded(
                statistics.fmean(all_amplitudes)
            ),
            "amp_std": rounded(
                statistics.pstdev(all_amplitudes)
            ),
            "amp_p10": rounded(
                percentile(all_amplitudes, .10)
            ),
            "amp_p90": rounded(
                percentile(all_amplitudes, .90)
            ),
            "subcarrier_profile_std": rounded(
                statistics.pstdev(mean_by_position)
            ),
            "temporal_spread": rounded(
                statistics.fmean(std_by_position)
            ),
            "change_mean": rounded(
                statistics.fmean(consecutive_changes)
            ),
            "change_p90": rounded(
                percentile(consecutive_changes, .90)
            ),
            "rssi_mean": rounded(
                statistics.fmean(rssi)
            ),
            "rssi_std": rounded(
                statistics.pstdev(rssi)
            ),
        })

    print(f"{path.name}: {len(windows)} windows")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=features[0].keys()
    )
    writer.writeheader()
    writer.writerows(features)

counts = {
    label: sum(row["label"] == label for row in features)
    for label in sorted(LABELS)
}

print(f"Total feature windows: {len(features)}")
print(f"Windows by class: {counts}")
print(f"Saved: {OUTPUT}")