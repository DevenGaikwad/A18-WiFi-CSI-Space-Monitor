"""CSI Space Monitor: read live_csi.py output without opening the serial port."""

import csv
import math
from collections import deque
from datetime import datetime, timezone
from html import escape
from pathlib import Path

import streamlit as st


ROOT = Path(__file__).resolve().parent
LIVE_DIR = ROOT / "data" / "live"
LABELS = ("NO_PRESENCE", "PERSON_PRESENT", "PERSON_MOVING")
SCORE_FIELDS = (
    "rf_score_no_presence",
    "rf_score_person_present",
    "rf_score_person_moving",
)
COLORS = ("#23e4a1", "#ffd15c", "#a985ff")


def read_predictions(path):
    """Read recent complete rows, safely skipping a row still being written."""
    result = []
    with path.open("r", newline="", encoding="utf-8", errors="replace") as file:
        for row in deque(csv.DictReader(file), maxlen=180):
            try:
                stamp = datetime.fromisoformat(row["prediction_utc"])
                label = row["prediction"]
                frames = int(row["frames"])
                scores = tuple(float(row[key]) for key in SCORE_FIELDS)
                rssi = float(row["rssi_mean_dbm"])
                if (
                    stamp.tzinfo is None
                    or label not in LABELS
                    or frames < 10
                    or not math.isfinite(rssi)
                    or any(not math.isfinite(x) or x < 0 or x > 1 for x in scores)
                ):
                    continue
                result.append(
                    {
                        "time": stamp,
                        "label": label,
                        "frames": frames,
                        "scores": scores,
                        "rssi": rssi,
                    }
                )
            except (KeyError, TypeError, ValueError):
                continue
    return result


def age_text(seconds):
    if seconds < 0:
        return "clock mismatch"
    if seconds < 60:
        return f"{int(seconds)} s ago"
    if seconds < 3600:
        return f"{int(seconds // 60)} min ago"
    if seconds < 86400:
        return f"{int(seconds // 3600)} h ago"
    return f"{int(seconds // 86400)} d ago"


def score_chart(rows):
    """Draw a vector chart from the same scores shown in the prediction log."""
    points = rows[-60:]
    left, right, top, bottom = 48, 868, 18, 174
    grid = []
    for value in (0, 0.25, 0.5, 0.75, 1):
        y = bottom - value * (bottom - top)
        grid.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" '
            'stroke="#283347" stroke-width="1"/>'
            f'<text x="5" y="{y + 4:.1f}" fill="#8da0b9" '
            f'font-size="11">{value:.2f}</text>'
        )
    lines = []
    for index, color in enumerate(COLORS):
        coords = []
        for i, row in enumerate(points):
            x = left + (right - left) * i / max(1, len(points) - 1)
            y = bottom - row["scores"][index] * (bottom - top)
            coords.append((x, y))
        if coords:
            geometry = " ".join(
                ("M" if i == 0 else "L") + f"{x:.1f},{y:.1f}"
                for i, (x, y) in enumerate(coords)
            )
            x, y = coords[-1]
            lines.append(
                f'<path d="{geometry}" fill="none" stroke="{color}" '
                'stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" '
                'vector-effect="non-scaling-stroke"/>'
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.6" fill="{color}"/>'
            )
    first = escape(points[0]["time"].astimezone().strftime("%H:%M:%S"))
    last = escape(points[-1]["time"].astimezone().strftime("%H:%M:%S"))
    return (
        '<svg class="trend-chart" viewBox="0 0 900 205" role="img" '
        'aria-label="Recent uncalibrated Random Forest class scores">'
        + "".join(grid + lines)
        + f'<text x="{left}" y="199" fill="#8da0b9" font-size="11">{first}</text>'
        + f'<text x="{right}" y="199" text-anchor="end" fill="#8da0b9" '
        f'font-size="11">{last}</text></svg>'
    )


def metric(title, value, detail, color):
    return (
        f'<div class="metric" style="--accent:{color}">'
        f'<div class="metric-name">{escape(title)}</div>'
        f'<div class="metric-value">{escape(str(value))}</div>'
        f'<div class="metric-detail">{escape(detail)}</div></div>'
    )


def event_table(rows):
    items = []
    for row in reversed(rows[-8:]):
        color = COLORS[LABELS.index(row["label"])]
        time = escape(row["time"].astimezone().strftime("%H:%M:%S"))
        label = escape(row["label"].replace("_", " "))
        items.append(
            f'<tr><td>{time}</td><td><span class="state-tag" '
            f'style="--accent:{color}">{label}</span></td>'
            f'<td>{row["frames"]}</td><td>{row["rssi"]:.1f} dBm</td></tr>'
        )
    return (
        '<div class="table-wrap"><table><thead><tr><th>TIME / LOCAL</th>'
        '<th>MODEL STATE</th><th>CSI FRAMES</th><th>MEAN RSSI</th></tr></thead>'
        '<tbody>' + "".join(items) + "</tbody></table></div>"
    )


st.set_page_config(
    page_title="CSI SPACE MONITOR",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """<style>
    :root {--base:#080b12;--panel:#101621;--panel2:#121b29;--line:#2a3548;
      --muted:#8da0b9;--text:#f4f7ff;--green:#23e4a1;--blue:#69abff;
      --yellow:#ffd15c;--purple:#a985ff;--red:#fb6886}
    html {scroll-behavior:smooth}
    .stApp,[data-testid="stAppViewContainer"] {background:var(--base);color:var(--text)}
    [data-testid="stHeader"],[data-testid="stSidebar"] {display:none}
    .block-container {max-width:none;width:calc(100% - 250px);margin:0 0 0 250px;
      padding:0 30px 56px;box-sizing:border-box}
    [data-testid="stMarkdownContainer"] p {margin:0}
    .csi-shell {font-family:Bahnschrift,"Segoe UI",Arial,sans-serif;color:var(--text)}
    .csi-shell * {box-sizing:border-box}
    .csi-shell a {color:inherit;text-decoration:none}
    .mono,.eyebrow,.micro,.nav-num,.topbar,.status,.metric-name,.metric-detail,
    .panel-meta,.section-kicker,.state-tag,.legend,.table-wrap th,.table-wrap td,
    .source-line,.alert,.footnote {
      font-family:Consolas,"Lucida Console",monospace}
    .condensed,.brand-name,.nav-label,.headline,.panel-title,.metric-value,.state-value {
      font-family:"Bahnschrift Condensed","Arial Narrow",Bahnschrift,"Segoe UI",sans-serif;
      font-stretch:condensed;font-weight:800;letter-spacing:.035em}
    .side-rail {position:fixed;left:0;top:0;bottom:0;width:250px;z-index:80;
      background:#0c1018;border-right:1px solid var(--line);padding:26px 14px;
      overflow-y:auto;display:flex;flex-direction:column}
    .brand {display:flex;gap:13px;align-items:center;padding:12px 13px 35px}
    .brand-icon {width:36px;height:36px;flex:none;color:var(--green)}
    .brand-name {font-size:20px;line-height:1.05;white-space:nowrap}
    .micro {font-size:9px;letter-spacing:.16em;color:var(--muted);margin-top:5px}
    .rail-caption {font:10px Consolas,monospace;color:var(--muted);
      letter-spacing:.17em;margin:10px 13px 13px}
    .nav-item {display:flex;justify-content:space-between;align-items:center;
      padding:15px 13px;margin-bottom:6px;border:1px solid transparent;
      border-radius:4px;color:#a5b4c9}
    .nav-item:hover {background:#121c28;border-color:#34485b}
    .nav-item.active {background:#11241f;border-color:#235f4b;
      color:var(--green);box-shadow:inset 3px 0 0 var(--green)}
    .nav-label {font-size:15px}
    .nav-num {font-size:10px;color:var(--muted)}
    .rail-bottom {margin-top:auto;border-top:1px solid var(--line);
      padding:19px 13px 2px;color:var(--muted);font-size:11px;line-height:1.8}
    .rail-bottom strong {display:block;color:var(--green);font:700 10px Consolas,monospace;
      letter-spacing:.09em;margin-bottom:9px}
    .topbar {height:74px;display:flex;align-items:center;justify-content:space-between;
      gap:14px;border-bottom:1px solid var(--line);font-size:11px;letter-spacing:.06em}
    .topbar b {color:var(--blue)} .topbar small {font-size:10px;color:var(--muted)}
    .top-chip {border:1px solid #325b46;color:var(--green);background:#10251d;
      padding:9px 12px;font-weight:700;white-space:nowrap}
    .main-head {padding:30px 0 28px}
    .section-kicker {color:var(--green);font-size:10px;letter-spacing:.15em;font-weight:700}
    .headline {font-size:clamp(31px,3vw,45px);line-height:1.08;margin:13px 0 12px;
      color:#f6f8fc;text-transform:uppercase}
    .headline .dot {color:var(--green)}
    .intro {color:#b2c2d6;font-size:14px;line-height:1.55}
    .live-strip {display:flex;align-items:center;gap:12px;flex-wrap:wrap;
      margin:0 0 19px;color:var(--muted);font:11px Consolas,monospace}
    .status {display:inline-flex;align-items:center;gap:9px;border:1px solid var(--accent);
      color:var(--accent);padding:7px 11px;font-size:10px;letter-spacing:.08em;font-weight:700}
    .status-dot {width:7px;height:7px;background:var(--accent);border-radius:50%}
    .source-line {color:var(--muted);font-size:10px;overflow-wrap:anywhere}
    .section-heading {display:flex;justify-content:space-between;align-items:baseline;
      gap:12px;margin:28px 0 11px}
    .section-heading h2 {color:#f4f7ff;font:800 19px "Bahnschrift Condensed","Arial Narrow",sans-serif;
      letter-spacing:.035em;margin:0;text-transform:uppercase}
    .section-heading span {font:10px Consolas,monospace;color:var(--muted);
      letter-spacing:.08em;text-transform:uppercase}
    .metric-grid {display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px}
    .metric {min-height:128px;background:var(--panel);border:1px solid var(--line);
      border-top:3px solid var(--accent);padding:17px 16px}
    .metric-name {color:#b6c9e5;font-size:11px;letter-spacing:.06em}
    .metric-value {color:var(--accent);font-size:clamp(22px,2vw,31px);
      line-height:1.1;margin:15px 0 9px;overflow-wrap:anywhere}
    .metric-detail {color:var(--muted);font-size:10px;line-height:1.4}
    .alert {border:1px solid #6f394a;border-left:3px solid var(--red);
      background:#26151d;color:#ffd5df;padding:13px 15px;margin:16px 0 0;
      font-size:11px;line-height:1.6}
    .alert b {color:var(--red);margin-right:10px}
    .panel {background:var(--panel);border:1px solid var(--line)}
    .panel-head {display:flex;justify-content:space-between;align-items:center;gap:10px;
      padding:15px 18px;border-bottom:1px solid var(--line)}
    .panel-title {font-size:15px;color:#fff;text-transform:uppercase}
    .panel-title::before {content:"";display:inline-block;width:6px;height:6px;
      background:var(--green);margin-right:11px;vertical-align:middle}
    .panel-meta {color:var(--muted);font-size:9px;letter-spacing:.09em;text-align:right}
    .signal-map {padding:25px 20px;background-image:linear-gradient(#17202d66 1px,transparent 1px),
      linear-gradient(90deg,#17202d66 1px,transparent 1px);
      background-size:24px 24px;display:grid;grid-template-columns:1fr 45px 1fr 45px 1fr 45px 1fr;
      align-items:center;min-height:160px}
    .flow-node {min-height:102px;border:1px solid #39536a;background:#101b2b;
      padding:14px 14px 11px;min-width:0}
    .flow-node:nth-of-type(1) {border-color:#26765a;background:#10221e}
    .signal-map .flow-node:last-child {border-color:#775aa9;background:#1b172a}
    .node-id {font:10px Consolas,monospace;color:var(--blue);letter-spacing:.12em}
    .node-main {font:800 19px "Bahnschrift Condensed","Arial Narrow",sans-serif;
      color:#f2f6ff;margin:8px 0 6px;text-transform:uppercase}
    .node-note {font:10px Consolas,monospace;color:#a8bed8;line-height:1.45}
    .flow-wire {height:2px;background:#59718b;position:relative}
    .flow-wire::after {content:"";position:absolute;right:-1px;top:-4px;
      border-left:6px solid #59718b;border-top:5px solid transparent;
      border-bottom:5px solid transparent}
    .panel-foot {padding:11px 17px;border-top:1px solid var(--line);
      color:var(--muted);font:10px Consolas,monospace}
    .score-grid {display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}
    .score {background:var(--panel);border:1px solid var(--line);border-left:3px solid var(--accent);
      padding:16px 17px 19px}
    .score-line {display:flex;align-items:baseline;justify-content:space-between;gap:8px}
    .score-name {color:#f4f7ff;font:800 15px "Bahnschrift Condensed","Arial Narrow",sans-serif}
    .score-number {color:var(--accent);font:700 21px Consolas,monospace}
    .score-track {height:6px;background:#2a3548;margin-top:17px}
    .score-fill {height:100%;width:var(--width);background:var(--accent)}
    .score-explain {margin:13px 0 0;color:var(--muted);font:11px Consolas,monospace}
    .trend-layout {display:grid;grid-template-columns:minmax(0,1.7fr) minmax(320px,1fr);gap:14px}
    .plot-body {padding:13px 14px 8px}
    .trend-chart {width:100%;height:auto;display:block;font-family:Consolas,monospace}
    .legend {display:flex;gap:20px;flex-wrap:wrap;border-top:1px solid var(--line);
      padding:12px 16px;color:#b9c9d9;font-size:10px}
    .legend i {display:inline-block;width:8px;height:8px;background:var(--accent);margin-right:6px}
    .table-wrap {overflow-x:auto}
    .table-wrap table {border-collapse:collapse;width:100%;font-size:10px}
    .table-wrap th,.table-wrap td {padding:11px 12px;text-align:left;border-bottom:1px solid #263041;
      white-space:nowrap}
    .table-wrap th {color:var(--muted);letter-spacing:.04em;font-size:9px}
    .table-wrap td {color:#d4e3f6}
    .state-tag {color:var(--accent);font-size:10px;font-weight:700}
    .footnote {margin:28px 0 0;color:var(--muted);font-size:10px;line-height:1.7;
      border-top:1px solid var(--line);padding-top:15px}
    @media(max-width:1100px) {
      .side-rail {width:195px}.block-container {margin-left:195px;width:calc(100% - 195px)}
      .metric-grid {grid-template-columns:repeat(2,minmax(0,1fr))}
      .trend-layout {grid-template-columns:1fr}.signal-map {grid-template-columns:repeat(4,minmax(0,1fr));gap:8px}
      .flow-wire {display:none}
    }
    @media(max-width:760px) {
      .side-rail {display:none}.block-container {margin-left:0;width:100%;padding:0 16px 40px}
      .metric-grid,.score-grid {grid-template-columns:1fr 1fr}
      .topbar small {display:none}.signal-map {grid-template-columns:1fr 1fr}
    }
    </style>""",
    unsafe_allow_html=True,
)

st.markdown(
    """<div class="csi-shell">
    <aside class="side-rail">
      <div class="brand">
        <svg class="brand-icon" viewBox="0 0 36 36" fill="none" aria-hidden="true">
          <path d="M8 8H28V28H8V8ZM8 18H28M18 8V28" stroke="currentColor" stroke-width="1.5"/>
          <rect x="5" y="5" width="6" height="6" fill="#0c1018" stroke="currentColor"/>
          <rect x="25" y="5" width="6" height="6" fill="#0c1018" stroke="currentColor"/>
          <rect x="5" y="25" width="6" height="6" fill="#0c1018" stroke="currentColor"/>
          <rect x="25" y="25" width="6" height="6" fill="#0c1018" stroke="currentColor"/>
        </svg>
        <div><div class="brand-name">CSI SPACE<br>MONITOR</div>
        <div class="micro">SENSING CONTROL ROOM</div></div>
      </div>
      <div class="rail-caption">ENGINEERING WORKSPACE</div>
      <a class="nav-item active" href="#overview"><span class="nav-label">OVERVIEW</span><span class="nav-num">01</span></a>
      <a class="nav-item" href="#signal-path"><span class="nav-label">SIGNAL PATH</span><span class="nav-num">02</span></a>
      <a class="nav-item" href="#class-scores"><span class="nav-label">CLASS SCORES</span><span class="nav-num">03</span></a>
      <a class="nav-item" href="#history"><span class="nav-label">HISTORY</span><span class="nav-num">04</span></a>
      <a class="nav-item" href="#evidence"><span class="nav-label">EVIDENCE</span><span class="nav-num">05</span></a>
      <div class="rail-bottom"><strong>● LOCAL ANALYSIS</strong>
        ESP32 CSI · Channel 06<br>Random Forest · 2 s windows<br>Three experimental states
      </div>
    </aside>
    <div class="topbar"><div><b>CSI</b> / SPACE MONITOR <small>&nbsp; / &nbsp; CONTROLLED EXPERIMENT</small></div>
      <span class="top-chip">● LOCAL DASHBOARD</span></div>
    </div>""",
    unsafe_allow_html=True,
)


@st.fragment(run_every="2s")
def show_monitor():
    error = ""
    path = None
    rows = []
    try:
        files = sorted(
            LIVE_DIR.glob("*_predictions.csv"),
            key=lambda p: (p.stat().st_mtime, p.name),
            reverse=True,
        ) if LIVE_DIR.exists() else []
        if files:
            path = files[0]
            rows = read_predictions(path)
    except OSError as exc:
        error = f"Prediction log could not be read: {exc}"

    latest = rows[-1] if rows else None
    age = (
        (datetime.now(timezone.utc) - latest["time"].astimezone(timezone.utc)).total_seconds()
        if latest else None
    )
    is_live = age is not None and 0 <= age <= 8
    status = "LIVE FEED" if is_live else ("RECORDED / STALE" if latest else "WAITING FOR DATA")
    status_color = "#23e4a1" if is_live else "#ffd15c"
    source = escape(path.name) if path else "No prediction CSV found"
    when = escape(latest["time"].astimezone().strftime("%d %b %Y · %H:%M:%S")) if latest else "—"
    label = latest["label"].replace("_", " ") if latest else "NO DATA"
    state_color = COLORS[LABELS.index(latest["label"])] if latest else "#8da0b9"
    frames = latest["frames"] if latest else "—"
    rssi = f'{latest["rssi"]:.1f} dBm' if latest else "—"
    age_display = age_text(age) if age is not None else "waiting for complete window"

    st.markdown(
        '<div class="csi-shell"><section id="overview" class="main-head">'
        '<div class="section-kicker">+ CONTACTLESS WI-FI CSI / EXPERIMENTAL SYSTEM</div>'
        '<h1 class="headline">CSI SPACE MONITOR<span class="dot">.</span></h1>'
        '<div class="intro">Observe receiver CSI windows and the three recorded model states '
        'in one local control view. The receiver script remains the sole owner of the COM port.</div>'
        '</section><div class="live-strip">'
        f'<span class="status" style="--accent:{status_color}"><i class="status-dot"></i>{status}</span>'
        f'<span>LAST WINDOW: {escape(age_display.upper())}</span>'
        '<span>REFRESH: 2 S</span>'
        f'<span class="source-line">SOURCE: {source}</span></div>'
        '<div class="section-heading"><h2>WINDOW TELEMETRY</h2>'
        '<span>RECORDED EVIDENCE / CURRENT FILE</span></div>'
        '<div class="metric-grid">'
        + metric("MODEL STATE", label, f"Latest window · {when}", state_color)
        + metric("CSI FRAMES", frames, "Valid frames / latest 2 s window", "#69abff")
        + metric("MEAN RSSI", rssi, "Received signal strength", "#ffd15c")
        + metric("RECENT WINDOWS", len(rows), "Valid rows shown / maximum 180", "#a985ff")
        + "</div></div>",
        unsafe_allow_html=True,
    )

    if error:
        st.markdown(
            '<div class="csi-shell alert"><b>READ ERROR</b>'
            + escape(error)
            + "</div>",
            unsafe_allow_html=True,
        )
    elif latest and not -83.2 <= latest["rssi"] <= -68.4:
        st.markdown(
            '<div class="csi-shell alert"><b>OUTSIDE REFERENCE RSSI</b>'
            'Latest mean RSSI is outside the training set’s central 1st–99th percentile '
            'range (−83.2 to −68.4 dBm). Interpret this model state cautiously. '
            'This warning is not a fourth person state.</div>',
            unsafe_allow_html=True,
        )
    elif not latest:
        st.markdown(
            '<div class="csi-shell alert"><b>WAITING</b>'
            'Start live_csi.py with the receiver COM port. The first complete '
            'two-second window will appear here.</div>',
            unsafe_allow_html=True,
        )

    tx_mac = "A4:F0:0F:69:1A:94"
    st.markdown(
        '<div class="csi-shell" id="signal-path">'
        '<div class="section-heading"><h2>SIGNAL PATH</h2>'
        '<span>TWO ESP32 NODES / LOCAL INFERENCE</span></div>'
        '<div class="panel"><div class="panel-head"><div class="panel-title">CSI ACQUISITION PATH</div>'
        '<div class="panel-meta">CONFIGURED TOPOLOGY / NOT A LIVE DEVICE-HEALTH CHECK</div></div>'
        '<div class="signal-map">'
        f'<div class="flow-node"><div class="node-id">01 / TRANSMITTER</div>'
        f'<div class="node-main">ESP32 TX</div><div class="node-note">MAC {tx_mac}</div></div>'
        '<div class="flow-wire"></div>'
        '<div class="flow-node"><div class="node-id">02 / RADIO PATH</div>'
        '<div class="node-main">WI-FI CH 06</div><div class="node-note">Controlled aisle geometry</div></div>'
        '<div class="flow-wire"></div>'
        '<div class="flow-node"><div class="node-id">03 / RECEIVER</div>'
        f'<div class="node-main">ESP32 RX</div><div class="node-note">{escape(str(frames))} CSI frames / window</div></div>'
        '<div class="flow-wire"></div>'
        '<div class="flow-node"><div class="node-id">04 / CLASSIFIER</div>'
        '<div class="node-main">RANDOM FOREST</div><div class="node-note">10 features · 3 classes</div></div>'
        '</div><div class="panel-foot">TX firmware → radio path → receiver CSI → '
        'two-second feature window → recorded prediction</div></div></div>',
        unsafe_allow_html=True,
    )

    score_boxes = []
    for i, name in enumerate(LABELS):
        score = latest["scores"][i] if latest else 0
        score_display = f"{score*100:.1f}%" if latest else "—"
        score_boxes.append(
            f'<div class="score" style="--accent:{COLORS[i]};--width:{score*100:.1f}%">'
            f'<div class="score-line"><span class="score-name">{name.replace("_", " ")}</span>'
            f'<span class="score-number">{score_display}</span></div>'
            '<div class="score-track"><div class="score-fill"></div></div></div>'
        )
    st.markdown(
        '<div class="csi-shell" id="class-scores"><div class="section-heading">'
        '<h2>CLASS SCORE INSPECTOR</h2><span>LATEST WINDOW / RANDOM FOREST</span></div>'
        '<div class="score-grid">' + "".join(score_boxes) + '</div>'
        '<div class="score-explain">Uncalibrated Random Forest class scores. '
        'These percentages are not verified probabilities of a person being present.</div></div>',
        unsafe_allow_html=True,
    )

    plot = score_chart(rows) if rows else '<div class="panel-foot">NO COMPLETE WINDOWS RECORDED YET</div>'
    table = event_table(rows) if rows else '<div class="panel-foot">NO RECENT PREDICTIONS</div>'
    st.markdown(
        '<div class="csi-shell" id="history"><div class="section-heading">'
        '<h2>RECORDED TIMELINE</h2><span>NO INTERPOLATED PREDICTIONS</span></div>'
        '<div class="trend-layout"><div class="panel"><div class="panel-head">'
        '<div class="panel-title">CLASS SCORE TREND</div>'
        '<div class="panel-meta">LAST 60 VALID WINDOWS</div></div>'
        f'<div class="plot-body">{plot}</div>'
        '<div class="legend">'
        '<span><i style="--accent:#23e4a1"></i>NO PRESENCE</span>'
        '<span><i style="--accent:#ffd15c"></i>PERSON PRESENT</span>'
        '<span><i style="--accent:#a985ff"></i>PERSON MOVING</span></div></div>'
        '<div class="panel"><div class="panel-head"><div class="panel-title">EVENT LOG</div>'
        '<div class="panel-meta">NEWEST FIRST / LAST 8</div></div>'
        + table + '</div></div></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="csi-shell footnote" id="evidence">'
        'CONTROLLED THREE-CLASS CSI DATASET · 45 RECORDINGS · 26,780 CSI ROWS · '
        'VALIDATION ON TRAINING ROUNDS 1–13: 78.39% · HELD-OUT ROUNDS 14–15: '
        '114/168 CORRECT WINDOWS (67.86%). LIVE PERFORMANCE REQUIRES SEPARATE '
        'GROUND-TRUTH LABELS. A STALE DISPLAY SHOWS SAVED DATA, NOT CURRENT OCCUPANCY.'
        '</div>',
        unsafe_allow_html=True,
    )


show_monitor()
