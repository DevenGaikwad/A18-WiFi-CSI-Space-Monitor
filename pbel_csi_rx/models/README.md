# Verified Baseline Model

lab_b_rf_rounds1_13.joblib is the unchanged model from the verified receiver archive. SHA-256: 76363de85761346f2b49f310803e71adbe25b229f8a11778d6c8f127acbd54d6.

It uses ten ordered features and two-second windows, fitted only on rounds 1–13. The original artifact reproduced 114/168 held-out predictions in the report audit. Its recorded local scikit-learn version was 1.8.0. Preserve this artifact when running training separately. Joblib/pickle loading can execute code; use this known-provenance artifact and do not substitute unknown files.
