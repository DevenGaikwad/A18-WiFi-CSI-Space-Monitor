# CSI SPACE MONITOR
The interface has Overview, Signal Path, Class Scores, History and Evidence navigation. Overview shows the current predicted state, valid CSI frames for the latest window, mean RSSI, recent rows and source CSV. Signal Path depicts configured modules and is not a health check. Class Scores are uncalibrated Random Forest scores. History plots 60 valid windows and shows the latest eight events; up to 180 rows are read. Evidence displays baseline dataset and evaluation scope.

The source uses `st.fragment(run_every="2s")`; a latest timestamp age between zero and eight seconds yields LIVE FEED. Older data are RECORDED / STALE. No complete row yields WAITING FOR DATA. The red RSSI warning is a reliability flag, not a fourth class. It must remain visible in documentary captures.

Run the live script first and dashboard second as in README. The dashboard never opens serial. A prediction time is stored in UTC and displayed in the host's local zone. In the provided evidence, 17:50:07 UTC equals 23:20:07 IST; both screens show NO_PRESENCE with 20 frames. S05 only shows Overview/Signal Path; the lower panels were verified from source, not claimed as separately captured screenshots.

The Python script and dashboard require serial/Streamlit dependencies as appropriate. Streamlit's exact installed version was not present in the older receiver ZIP. The requirement range supports the fragment API and is not an original environment lock.

The current live parser requires a 128-byte record and 256 hex characters. It accepts invalid-word flags 0 or 1, but always drops four bytes. Training equivalence was checked on the recorded flag-1 format. A changed packet format needs separate compatibility validation.
