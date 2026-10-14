"""Reproduce Group A18 LAB_B results and plain Matplotlib figures.

Only the unmodified 45-session archive is used. Every row is checked before
feature construction. The split is by full recording round (1-13 / 14-15).
"""

from __future__ import annotations

import csv
import io
import json
import zipfile
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import LeaveOneGroupOut

CLASSES = ["NO_PRESENCE", "PERSON_PRESENT", "PERSON_MOVING"]
FEATURES = [
    "amp_mean", "amp_std", "amp_p10", "amp_p90", "subcarrier_profile_std",
    "temporal_spread", "change_mean", "change_p90", "rssi_mean", "rssi_std",
]


def rf():
    return RandomForestClassifier(
        n_estimators=300, min_samples_leaf=3, class_weight="balanced",
        random_state=42, n_jobs=-1,
    )


def run(archive_path: str | Path, output_dir: str | Path):
    archive_path = Path(archive_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples, sessions = [], []
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise ValueError("Archive CRC check failed")
        names = sorted(name for name in archive.namelist() if name.lower().endswith(".csv"))
        if len(names) != 45:
            raise ValueError(f"Expected 45 labelled sessions, found {len(names)}")

        for j, name in enumerate(names):
            round_id = j // 3 + 1
            rows = list(csv.DictReader(io.StringIO(archive.read(name).decode("utf-8-sig"))))
            if not rows:
                raise ValueError(f"Empty recording: {name}")
            label, session_id = rows[0]["label"], rows[0]["session_id"]
            if label not in CLASSES or any(
                r["label"] != label or r["session_id"] != session_id
                or r["tx_mac"].lower() != "a4:f0:0f:69:1a:94"
                or r["channel"] != "6" or int(r["csi_len"]) != 128
                or r["first_word_invalid"] != "1" or len(r["csi_hex"]) != 256
                for r in rows
            ):
                raise ValueError(f"Bad/unexpected recording: {name}")

            sessions.append({"round": round_id, "label": label, "session_id": session_id,
                             "file": name, "frames": len(rows)})
            bins = [[] for _ in range(28)]
            for r in rows:
                # 28 non-overlapping 2 s windows: receiver timestamps [20,76) s.
                k = (int(r["rx_timestamp_us"]) - 20_000_000) // 2_000_000
                if 0 <= k < 28:
                    signed = np.frombuffer(bytes.fromhex(r["csi_hex"]), dtype=np.int8).astype(float)
                    # Skip two invalid complex samples (first_word_invalid=1).
                    amp = np.hypot(signed[4::2], signed[5::2])
                    bins[k].append((amp, float(r["rssi_dbm"])))
            for k, frames in enumerate(bins):
                if len(frames) < 10:
                    raise ValueError(f"Fewer than 10 frames in {name}, window {k}")
                a = np.asarray([f[0] for f in frames])
                rssi = np.asarray([f[1] for f in frames])
                delta = np.abs(np.diff(a, axis=0))
                flat = a.ravel()
                values = [flat.mean(), flat.std(), np.percentile(flat, 10),
                          np.percentile(flat, 90), a.mean(axis=0).std(),
                          a.std(axis=0).mean(), delta.mean(), np.percentile(delta, 90),
                          rssi.mean(), rssi.std()]
                samples.append({"round": round_id, "label": label, "session_id": session_id,
                                "window": k, "frames": len(frames), **dict(zip(FEATURES, values))})

    sess = pd.DataFrame(sessions)
    d = pd.DataFrame(samples)
    for rid in range(1, 16):
        if Counter(sess[sess["round"] == rid].label) != Counter({c: 1 for c in CLASSES}):
            raise ValueError(f"Round {rid} lacks one class")
        if Counter(d[d["round"] == rid].label) != Counter({c: 28 for c in CLASSES}):
            raise ValueError(f"Round {rid} lacks 28 windows for each class")
    tr = d[d["round"] <= 13].copy().reset_index(drop=True)
    te = d[d["round"] >= 14].copy().reset_index(drop=True)
    oof = np.empty(len(tr), dtype=object)
    for a, b in LeaveOneGroupOut().split(tr[FEATURES], tr.label, tr["round"]):
        oof[b] = rf().fit(tr.iloc[a][FEATURES], tr.iloc[a].label).predict(tr.iloc[b][FEATURES])
    model = rf().fit(tr[FEATURES], tr.label)
    pred = model.predict(te[FEATURES])
    tr["pred"] = oof
    te["pred"] = pred
    cm_cv = confusion_matrix(tr.label, oof, labels=CLASSES)
    cm_te = confusion_matrix(te.label, pred, labels=CLASSES)
    assert cm_cv.tolist() == [[239, 64, 61], [42, 311, 11], [44, 14, 306]]
    assert cm_te.tolist() == [[26, 7, 23], [15, 32, 9], [0, 0, 56]]

    def save(name):
        plt.tight_layout()
        plt.savefig(output_dir / name, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close()

    # Figures use standard sklearn/Matplotlib components on a white background.
    for cm, title, name in [
        (cm_cv, "Leave-one-round-out validation (rounds 1-13)", "01_validation_confusion.png"),
        (cm_te, "Unseen test: rounds 14-15 (168 windows)", "02_test_confusion.png"),
    ]:
        fig, ax = plt.subplots(figsize=(6.6, 4.5))
        ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(ax=ax, cmap="Blues", values_format="d",
                                                           colorbar=False, xticks_rotation=25)
        ax.set_title(title)
        save(name)

    scores = [accuracy_score(part.label, part.pred) for part in
              [tr[tr["round"] == i] for i in range(1, 14)] +
              [te[te["round"] == i] for i in (14, 15)]]
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    ax.bar(range(1, 14), np.array(scores[:13]) * 100, color="0.55", label="Validation")
    ax.bar([14, 15], np.array(scores[13:]) * 100, color="0.15", label="Held-out test")
    ax.axvline(13.5, color="0.3", linestyle="--", linewidth=1)
    ax.set(xlabel="Complete recording round", ylabel="Correct windows (%)", ylim=(0, 100),
           title="Accuracy by round (separate sessions)")
    ax.set_xticks(range(1, 16))
    ax.legend()
    save("03_round_accuracy.png")

    # Aggregate 28 related windows to each independent training session.
    session = tr.groupby(["session_id", "label"], as_index=False)[["rssi_mean", "change_p90"]].mean()
    fig, axes = plt.subplots(1, 2, figsize=(9.1, 3.6))
    for ax, key, ylabel in zip(axes, ["rssi_mean", "change_p90"],
                                ["Mean RSSI (dBm)", "90th percentile of CSI changes"]):
        values = [session[session.label == label][key].to_numpy() for label in CLASSES]
        ax.boxplot(values, showfliers=True)
        for j, vals in enumerate(values, 1):
            ax.scatter(np.full(len(vals), j), vals, color="black", alpha=0.35, s=12)
        ax.set_xticks([1, 2, 3], ["No presence", "Stationary", "Moving"], rotation=14)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Training session averages (13 sessions per class)")
    save("04_session_feature_overlap.png")

    corr_features = [key for key in FEATURES if tr[key].nunique() > 1]
    corr = tr[corr_features].corr()
    fig, ax = plt.subplots(figsize=(7.1, 6))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(corr)), corr.columns, rotation=65, ha="right")
    ax.set_yticks(range(len(corr)), corr.index)
    fig.colorbar(im, ax=ax, shrink=.8, label="Pearson correlation")
    ax.set_title("Training-window feature correlation")
    save("05_feature_correlation.png")

    importance = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(importance.index, importance.values)
    ax.set(xlabel="Mean decrease in impurity (training data)",
           title="Random Forest feature importance; descriptive only")
    save("06_feature_importance.png")

    print(f"CSV files: {len(sess)}; CSI rows: {int(sess.frames.sum())}")
    print(f"Training windows: {len(tr)}; test windows: {len(te)}")
    print(f"Leave-one-round-out accuracy: {accuracy_score(tr.label, oof):.4f}")
    print(f"Leave-one-round-out macro F1: {f1_score(tr.label, oof, average='macro'):.4f}")
    print(f"Held-out test accuracy: {accuracy_score(te.label, pred):.4f}")
    print(f"Held-out test macro F1: {f1_score(te.label, pred, average='macro'):.4f}")
    print("Held-out confusion matrix (actual rows, predicted columns):")
    print(cm_te)
    print(classification_report(te.label, pred, labels=CLASSES, zero_division=0, digits=3))
    summary = {"archive": archive_path.name, "csv_sessions": len(sess), "csi_rows": int(sess.frames.sum()),
               "windows": len(d), "train_rounds": list(range(1, 14)), "test_rounds": [14, 15],
               "training_windows": len(tr), "test_windows": len(te),
               "validation_accuracy": float(accuracy_score(tr.label, oof)),
               "validation_macro_f1": float(f1_score(tr.label, oof, average="macro")),
               "test_accuracy": float(accuracy_score(te.label, pred)),
               "test_macro_f1": float(f1_score(te.label, pred, average="macro")),
               "cm_validation": cm_cv.tolist(), "cm_test": cm_te.tolist(),
               "round_scores": scores,
               "report_test": classification_report(te.label, pred, labels=CLASSES,
                                                    output_dict=True, zero_division=0),
               "test_session_votes": [{"round": int(x["round"].iloc[0]),
                                       "actual": x.label.iloc[0],
                                       "predicted": Counter(x.pred).most_common(1)[0][0]}
                                      for _, x in te.groupby("session_id", sort=False)],
               "feature_names": FEATURES,
               "constant_train_features": [k for k in FEATURES if tr[k].nunique() == 1],
               "feature_importances": dict(zip(FEATURES, model.feature_importances_))}
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary




if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("archive", type=Path)
    p.add_argument("--out", type=Path, default=Path("results/reproduced"))
    args = p.parse_args()
    run(args.archive, args.out)
