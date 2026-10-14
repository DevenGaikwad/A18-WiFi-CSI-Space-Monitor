import argparse
import csv
import time
from datetime import datetime, timezone
from pathlib import Path

import serial

TX_MAC = "a4:f0:0f:69:1a:94"
LABELS = ("NO_PRESENCE", "PERSON_PRESENT", "PERSON_MOVING")

parser = argparse.ArgumentParser()
parser.add_argument("--label", required=True, choices=LABELS)
parser.add_argument("--seconds", type=int, default=60)
parser.add_argument("--delay", type=int, default=15)
parser.add_argument("--port", default="COM4")
args = parser.parse_args()

if args.seconds <= 0 or args.delay < 0:
    parser.error("--seconds must be positive and --delay cannot be negative")

session_id = (
    datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    + "_" + args.label
)
output_dir = Path(__file__).resolve().parent / "data" / "raw" / "LAB_B"
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / f"{session_id}.csv"

header = [
    "capture_utc", "session_id", "label", "rx_timestamp_us",
    "tx_mac", "rssi_dbm", "channel", "sig_mode",
    "csi_len", "first_word_invalid", "csi_hex",
]

saved = 0
invalid_rows = 0
last_status = ""

with serial.Serial(args.port, 115200, timeout=1) as board:
    # Opening COM4 may restart the ESP32. Let it boot, then discard boot text.
    time.sleep(3)
    board.reset_input_buffer()

    print(f"Label: {args.label}")
    print(f"Get into position now. Recording starts in {args.delay} seconds.")
    time.sleep(args.delay)
    board.reset_input_buffer()  # Discard frames received during preparation.
    print(f"Recording for {args.seconds} seconds...")

    deadline = time.monotonic() + args.seconds

    with output_path.open("x", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)

        while time.monotonic() < deadline:
            line = board.readline().decode("ascii", errors="replace").strip()

            if "RX_STATUS" in line:
                last_status = line
                continue
            if not line.startswith("CSI_DATA,"):
                continue

            parts = line.split(",", 8)
            if len(parts) != 9:
                invalid_rows += 1
                continue

            _, rx_time, mac, rssi, channel, mode, length, first_bad, hex_data = parts

            try:
                length_num = int(length)
                if (
                    mac.lower() != TX_MAC
                    or int(channel) != 6
                    or length_num <= 0
                    or length_num > 256
                    or len(hex_data) != 2 * length_num
                ):
                    raise ValueError("Unexpected CSI frame")

                bytes.fromhex(hex_data)
                row = [
                    datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                    session_id, args.label, int(rx_time), mac.lower(),
                    int(rssi), int(channel), int(mode),
                    length_num, int(first_bad), hex_data,
                ]
            except ValueError:
                invalid_rows += 1
                continue

            writer.writerow(row)
            file.flush()
            saved += 1

print(f"Saved: {saved} CSI rows")
print(f"Rejected malformed/unexpected rows: {invalid_rows}")
print(f"Last receiver status: {last_status or 'not seen'}")
print(f"CSV: {output_path}")