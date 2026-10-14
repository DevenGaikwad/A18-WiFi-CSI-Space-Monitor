# Model evaluation
Ten inputs, in order: amp_mean, amp_std, amp_p10, amp_p90, subcarrier_profile_std, temporal_spread, change_mean, change_p90, rssi_mean, rssi_std. The forest uses 300 trees, min_samples_leaf=3, balanced class weights, random_state=42 and n_jobs=-1.

Leave-one-round-out validation within rounds 1–13: accuracy 0.7838827839, macro F1 0.7815287104. Final fit on these rounds: training accuracy 0.9725274725. Fixed held-out rounds 14–15: accuracy 0.6785714286, macro F1 0.6625148208, matrix [[26,7,23],[15,32,9],[0,0,56]]. Class order is NO_PRESENCE, PERSON_PRESENT, PERSON_MOVING; rows actual, columns predicted. Session majority was correct for five of six held-out recordings. Round 15 stationary presence was misclassified as NO_PRESENCE.

The report audit reran the fetched notebook's evaluation module and matched every supplied metrics.json value. The original saved model also matched the test matrix. Baseline artifacts were preserved.

Exploratory ablation, performed during report preparation with the same split and parameters: removing rssi_mean/rssi_std gives validation 62.82%, held-out 61.90%, held-out macro F1 0.6061. It is a post-baseline analysis on already inspected test rounds, not a new untouched test or a basis for tuning the model. RSSI-related impurity importance totals about 42.79%; this is not proof of causal influence. `amp_p10` is zero throughout training because more than 10% of retained positions are zero in every recorded frame.

No live accuracy, latency benchmark or cross-room performance was measured from the supplied evidence. The dashboard's -1.3 dBm example is far outside the verified training-window reference range [-83.209,-68.391] dBm. The latest underlying CSV/raw stream is absent, so the cause remains unresolved.
