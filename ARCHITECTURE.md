# Implemented architecture
ESP32 TX schedules a four-byte sequence broadcast every 20 ms on channel 6, HT20, MCS0 LGI. ESP32 RX filters the sender MAC and copies every fifth eligible CSI callback to a 16-entry FreeRTOS queue. A separate firmware task prints CSI_DATA lines at the configured 115200-baud UART setting.

Python decodes 128 signed CSI bytes, drops the first four, forms 62 magnitudes, constructs two-second windows and calculates ten features. The saved Random Forest predicts three class labels and uncalibrated class scores. Predictions are flushed to UTC-named CSV files. Streamlit reads the newest file, validates complete rows and refreshes every two seconds.

The host design uses separate acquisition/inference and dashboard processes. `live_csi.py` has a synchronous serial loop; it does not define a custom Python ML worker thread. `n_jobs=-1` permits scikit-learn parallel computation. RX callback queueing is firmware-side concurrency. No external router or on-ESP32 ML is implemented.
