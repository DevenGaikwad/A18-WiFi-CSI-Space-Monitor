# Hardware setup
Actual TX and RX identify in archived flash logs as ESP32-D0WD-V3 revision 3.1. The preparation record assigns TX to COM3 with CH9102 and RX to COM4 with CP2102. USB bridge assignments come from the team's setup record; confirm actual ports before flashing. The original ESP32-S3 proposal was implemented using these classic ESP32 boards.

Use ESP-IDF v5.5.5 and target `esp32` for both projects. In an ESP-IDF terminal opened in each project:
```powershell
idf.py set-target esp32
idf.py build
```
Then flash only the correct board and port:
```powershell
idf.py -p COM3 flash monitor
```
Use COM4 in the receiver folder. Press Ctrl+] to close IDF monitor before running `live_csi.py`. A reset/BOOT press alone does not erase firmware; recover actual connection failures before attempting a flash. These are documented start steps, not a fresh hardware rebuild performed for this report.

The sender MAC filter is in RX `main/main.c` and Python. Preserve its actual value for the demonstrated hardware; if a board is replaced, re-identify its MAC and record that change. Do not treat the configured topology as a live device-health test. Physical distance was reported as approximately 2–2.5 m; no measured setup photograph or controlled 1/2/3 m comparison was supplied.
