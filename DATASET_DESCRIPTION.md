# Recorded dataset
The 6 October pilot contains 15 sessions and 11,041 raw rows. A separate timing trim keeps receiver uptime >=20 s, leaving 8,461 rows. Pilot files remain separate from the evaluated LAB_B dataset.

LAB_B, 9 October: 45 CSV files; NO_PRESENCE 8,941 rows, PERSON_PRESENT 8,916, PERSON_MOVING 8,923; total 26,780. Fifteen chronologically sorted groups of three each contain one session per label. Every row has channel 6, signal mode 1, length 128, first_word_invalid=1 and 256 hexadecimal characters.

The fields are capture_utc, session_id, label, rx_timestamp_us, tx_mac, rssi_dbm, channel, sig_mode, csi_len, first_word_invalid and csi_hex. Receiver timestamps are microsecond uptime; laptop times are UTC. The MAC is separate metadata, not part of the 128 CSI bytes.

Each session supplies 28 non-overlapping two-second windows over [20 s,76 s). Windows have at least 10 frames; observed LAB_B counts are 17–21. Rounds 1–13 provide 1,092 development windows; rounds 14–15 supply 168 fixed test windows. Recordings and windows share session conditions; neither window count nor separate filenames establishes independent people or rooms.

Raw archives, live logs and personal experiment metadata are omitted from this source package. Obtain them under team-controlled access. Dataset hashes and labels are recorded in the report's supporting evidence, rather than an unrestricted public data release.
