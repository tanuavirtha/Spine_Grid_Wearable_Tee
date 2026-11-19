import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import json
import threading
import time
from collections import deque

import joblib
import numpy as np
import pandas as pd
import serial
from serial.tools import list_ports
from flask import Flask, jsonify, render_template_string

from posture_features import (
    IMUS, AX, GX, FLEX_COLS, WINDOW_CHANNELS, STATS, FS,
    WINDOW_ROWS, HOP_ROWS, absolute_angles, wrap180, flex_valid,
)

BAUD_RATE = 115200
SERIAL_PORT = None  # Set to "/dev/cu.usbmodemXXXX" if auto-detection fails.
CALIBRATION_SECONDS = 4.0
CALIBRATION_SKIP_SECONDS = 2.0
SAMPLE_INTERVAL_MS = 1000 / FS
HISTORY_LIMIT = 1800  # Keep at most ~72 seconds at 25 Hz.

BASE_DIR = __import__("pathlib").Path(__file__).resolve().parent.parent
ACTIVITY_PATH = BASE_DIR / "models" / "activity_model.joblib"
POSTURE_PATH = BASE_DIR / "models" / "posture_model.joblib"
CONFIG_PATH = BASE_DIR / "models" / "warning_config.json"

activity_artifact = joblib.load(ACTIVITY_PATH)
posture_artifact = joblib.load(POSTURE_PATH)
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = json.load(f)

activity_model = activity_artifact["model"]
posture_model = posture_artifact["model"]
activity_features = list(activity_artifact["features"])
posture_features = list(posture_artifact["features"])
tau = float(posture_artifact.get("tau", cfg.get("tau_poor", 0.5)))

app = Flask(__name__)
lock = threading.Lock()
latest_reading = None
last_received = 0.0
serial_status = "Searching for ESP32..."
history = deque(maxlen=HISTORY_LIMIT)
session_start = None
calibration_complete = False
calibration_baseline = None
calibration_message = "Keep still in your normal upright posture for calibration."
latest_prediction = {
    "activity": "Calibrating",
    "posture": "Waiting",
    "poor_probability": None,
    "warning": "Waiting",
    "model_note": "Experimental — trained on synthetic data",
    "window_ready": False,
    "sample_count": 0,
}
last_prediction_count = 0
poor_hold_seconds = 0.0
warning_state = "none"


def find_esp32_port():
    ports = list(list_ports.comports())
    for port in ports:
        device = port.device.lower()
        description = (port.description or "").lower()
        if (
            "usbmodem" in device or "usbserial" in device
            or "esp32" in description or "cp210" in description
            or "ch340" in description
        ):
            return port.device
    print("Could not automatically identify the ESP32 port. Available ports:")
    for port in ports:
        print(f"  {port.device}: {port.description}")
    return None


def row_from_reading(data, timestamp_ms, is_calibration):
    """Convert the ESP32 JSON payload to the repository's expected raw columns."""
    row = {
        "session_id": "live_session",
        "timestamp_ms": int(timestamp_ms),
        "is_calibration": int(is_calibration),
        "flex_lr_raw": float(data.get("flex1", np.nan)),
        "flex_rl_raw": float(data.get("flex2", np.nan)),
    }
    for name in ("upper", "left", "right"):
        vals = data.get(name)
        if not isinstance(vals, list) or len(vals) < 6:
            raise ValueError(f"{name} must contain ax, ay, az, gx, gy, gz")
        prefix = f"imu_{name}"
        for i, axis in enumerate(AX):
            row[f"{prefix}_{axis}"] = float(vals[i])
        for i, axis in enumerate(GX):
            row[f"{prefix}_{axis}"] = float(vals[i + 3])
    return row


def make_baseline(df):
    cal = df[df["is_calibration"] == 1].copy()
    if cal.empty:
        return None
    first_t = int(cal["timestamp_ms"].min())
    cal = cal[cal["timestamp_ms"] >= first_t + CALIBRATION_SKIP_SECONDS * 1000]
    if len(cal) < max(5, FS // 2):
        return None
    angles = absolute_angles(cal)
    base = {key: float(np.nanmean(values)) for key, values in angles.items()}
    for col in FLEX_COLS:
        values = cal[col].to_numpy(float)
        valid_values = np.where(flex_valid(values), values, np.nan)
        base[col] = float(np.nanmean(valid_values)) if np.isfinite(valid_values).any() else np.nan
    if not all(np.isfinite(v) for v in base.values()):
        return None
    return base


def derive_live_features(df, baseline):
    """Same derived-channel definitions as posture_features.py, using the live calibration."""
    angles = absolute_angles(df)
    d = pd.DataFrame(angles, index=df.index)
    d["upper_pitch_dev_deg"] = wrap180(d["upper_pitch_deg"] - baseline["upper_pitch_deg"])
    d["upper_roll_dev_deg"] = wrap180(d["upper_roll_deg"] - baseline["upper_roll_deg"])
    left_roll_dev = wrap180(d["left_shoulder_roll_deg"] - baseline["left_shoulder_roll_deg"])
    right_roll_dev = wrap180(d["right_shoulder_roll_deg"] - baseline["right_shoulder_roll_deg"])
    left_pitch_dev = wrap180(d["left_shoulder_pitch_deg"] - baseline["left_shoulder_pitch_deg"])
    right_pitch_dev = wrap180(d["right_shoulder_pitch_deg"] - baseline["right_shoulder_pitch_deg"])
    d["shoulder_asymmetry_dev_deg"] = np.abs(left_roll_dev - right_roll_dev)
    d["shoulder_rounding_dev_deg"] = (left_pitch_dev + right_pitch_dev) / 2 - d["upper_pitch_dev_deg"]

    for raw_col, dev_col in (("flex_lr_raw", "flex_lr_dev"), ("flex_rl_raw", "flex_rl_dev")):
        values = df[raw_col].to_numpy(float)
        d[dev_col] = np.where(
            flex_valid(values) & np.isfinite(baseline[raw_col]),
            values - baseline[raw_col],
            np.nan,
        )
    d["flex_dev_total"] = np.abs(d["flex_lr_dev"]) + np.abs(d["flex_rl_dev"])
    accel = df[[f"imu_upper_{axis}" for axis in AX]].to_numpy(float)
    gyro = df[[f"imu_upper_{axis}" for axis in GX]].to_numpy(float)
    d["accel_magnitude"] = np.sqrt(np.sum(accel ** 2, axis=1))
    d["gyro_magnitude"] = np.sqrt(np.sum(gyro ** 2, axis=1))
    d["thoracic_dev_deg"] = np.abs(d["upper_pitch_dev_deg"])
    return d


def window_vector(window_df, derived_df, artifact_features):
    combined = pd.concat([window_df.reset_index(drop=True), derived_df.reset_index(drop=True)], axis=1)
    arr = combined[WINDOW_CHANNELS].to_numpy(float)
    with np.errstate(all="ignore"):
        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0)
        low = np.nanmin(arr, axis=0)
        high = np.nanmax(arr, axis=0)
    values = np.column_stack([mean, std, low, high, high - low]).reshape(-1)
    names = [f"{channel}__{stat}" for channel in WINDOW_CHANNELS for stat in STATS]
    vector = pd.DataFrame([values], columns=names)
    missing = [name for name in artifact_features if name not in vector.columns]
    if missing:
        raise ValueError(f"Feature mismatch; missing: {missing[:8]}")
    return vector.loc[:, artifact_features]


def severity_score(X, config):
    refs = config.get("severity_reference_deg", {})
    scores = []
    for col, ref in refs.items():
        if col in X.columns:
            try:
                denom = float(ref)
                if denom > 0:
                    scores.append(abs(float(X.iloc[0][col])) / denom)
            except (TypeError, ValueError):
                pass
    return max(scores, default=0.0)


def update_prediction():
    global latest_prediction, last_prediction_count, poor_hold_seconds, warning_state
    if not calibration_complete or calibration_baseline is None:
        latest_prediction.update({
            "activity": "Calibrating",
            "posture": "Waiting",
            "poor_probability": None,
            "warning": "Waiting",
            "window_ready": False,
            "sample_count": len(history),
        })
        return
    if len(history) < WINDOW_ROWS:
        latest_prediction.update({
            "activity": "Collecting window",
            "posture": "Waiting",
            "poor_probability": None,
            "warning": "Waiting",
            "window_ready": False,
            "sample_count": len(history),
        })
        return
    # Predict once per hop (25 new samples = approximately 1 second).
    if len(history) - last_prediction_count < HOP_ROWS:
        return
    frame = pd.DataFrame(list(history))
    raw_window = frame.tail(WINDOW_ROWS).reset_index(drop=True)
    derived_all = derive_live_features(frame, calibration_baseline)
    derived_window = derived_all.tail(WINDOW_ROWS).reset_index(drop=True)
    try:
        X_activity = window_vector(raw_window, derived_window, activity_features)
        activity = str(activity_model.predict(X_activity)[0])
        X_posture = X_activity.copy()
        for act in cfg.get("applicable_activities", ["sitting", "standing", "walking"]):
            X_posture[f"ctx_{act}"] = float(activity == act)
        missing = [col for col in posture_features if col not in X_posture.columns]
        if missing:
            raise ValueError(f"Missing posture context/features: {missing}")
        X_posture = X_posture.loc[:, posture_features]
        p_poor = None
        posture = "not applicable"
        applicable = cfg.get("applicable_activities", ["sitting", "standing", "walking"])
        if activity in applicable:
            classes = list(posture_model.classes_)
            poor_idx = classes.index("poor")
            p_poor = float(posture_model.predict_proba(X_posture)[0, poor_idx])
            posture = "poor" if p_poor >= tau else "good"

        # Approximate persistence logic from the repository configuration.
        if activity in applicable and p_poor is not None and p_poor >= tau:
            poor_hold_seconds += HOP_ROWS / FS
        else:
            poor_hold_seconds = 0.0
        hold_s = float(cfg.get("hold_s", 5.0))
        critical_s = float(cfg.get("critical_sustain_s", hold_s))
        sev = severity_score(X_activity, cfg)
        if poor_hold_seconds >= hold_s:
            warning_state = "critical" if (
                poor_hold_seconds >= critical_s
                or sev >= float(cfg.get("critical_severity", 2.0))
            ) else "advisory"
        elif p_poor is not None and p_poor >= tau:
            warning_state = "advisory"
        else:
            warning_state = "none"

        latest_prediction = {
            "activity": activity,
            "posture": posture,
            "poor_probability": p_poor,
            "warning": warning_state,
            "model_note": "Experimental — trained on synthetic data",
            "window_ready": True,
            "sample_count": len(history),
            "threshold": tau,
            "severity_score": round(sev, 3),
        }
        last_prediction_count = len(history)
    except Exception as exc:
        latest_prediction.update({
            "activity": "Inference error",
            "posture": str(exc),
            "poor_probability": None,
            "warning": "Unavailable",
            "window_ready": False,
            "sample_count": len(history),
        })
        print("Inference error:", repr(exc))


def serial_reader():
    global latest_reading, last_received, serial_status
    global session_start, calibration_complete, calibration_baseline, calibration_message
    while True:
        port = SERIAL_PORT or find_esp32_port()
        if not port:
            with lock:
                serial_status = "ESP32 port not found"
            time.sleep(3)
            continue
        try:
            print(f"Opening ESP32 serial port: {port}")
            with serial.Serial(port, BAUD_RATE, timeout=1) as connection:
                time.sleep(2)
                connection.reset_input_buffer()
                with lock:
                    serial_status = f"Connected: {port}"
                print("Waiting for JSON sensor readings...")
                while True:
                    line = connection.readline().decode("utf-8", errors="ignore").strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    required = {"upper", "left", "right", "flex1", "flex2"}
                    if not required.issubset(data):
                        continue
                    try:
                        now = time.time()
                        with lock:
                            if session_start is None:
                                session_start = now
                            elapsed = now - session_start
                            is_cal = int(elapsed < CALIBRATION_SECONDS)
                            row = row_from_reading(data, int(elapsed * 1000), is_cal)
                            latest_reading = data
                            last_received = now
                            history.append(row)
                            if not calibration_complete and elapsed >= CALIBRATION_SECONDS:
                                baseline = make_baseline(pd.DataFrame(list(history)))
                                if baseline is not None:
                                    calibration_baseline = baseline
                                    calibration_complete = True
                                    calibration_message = "Calibration complete"
                                    print("Calibration complete; live inference enabled.")
                                else:
                                    calibration_message = "Calibration readings invalid; check flex sensors and restart."
                            update_prediction()
                    except Exception as exc:
                        print("Sensor processing error:", repr(exc))
        except (serial.SerialException, OSError) as error:
            with lock:
                serial_status = f"Serial disconnected: {error}"
            print(f"Serial error: {error}")
            time.sleep(3)


HTML = r"""
<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Spine-Grid | Live Posture Monitor</title>
<style>
*{box-sizing:border-box}body{font-family:Arial,sans-serif;background:#f3f5f9;color:#202938;margin:0;padding:24px}
main{max-width:1100px;margin:auto}h1{margin-bottom:5px}.muted{color:#687386;font-size:.92rem}
.status,.card{background:#fff;border-radius:12px;padding:16px;box-shadow:0 2px 8px #0000000a}
.status{margin:16px 0}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px;margin:14px 0}
.card h2{font-size:1.05rem;margin:0 0 12px}.big{font-size:1.5rem;font-weight:bold;margin:8px 0;overflow-wrap:anywhere}
.axis{display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #edf0f5;font-variant-numeric:tabular-nums}
.pill{display:inline-block;border-radius:999px;padding:5px 10px;background:#e9edf3;font-weight:bold}
#warning{font-size:1.15rem}pre{background:#202938;color:#e7f1ff;padding:14px;border-radius:10px;overflow-wrap:anywhere;white-space:pre-wrap}
</style></head><body><main>
<h1>Spine-Grid</h1><p class="muted">Live wearable sensor and experimental ML monitor</p>
<div class="status">Connection: <strong id="status">Connecting...</strong><br><span class="muted" id="updated">Waiting for sensor data</span><br><span class="muted" id="calibration">Calibration pending</span></div>
<div class="grid">
<div class="card"><h2>Activity</h2><div class="big" id="activity">Waiting</div><span class="pill">Model A</span></div>
<div class="card"><h2>Posture</h2><div class="big" id="posture">Waiting</div><div id="probability" class="muted">Poor-posture probability: --</div><span class="pill">Model B</span></div>
<div class="card"><h2>Warning</h2><div class="big" id="warning">Waiting</div><div class="muted" id="severity">Severity: --</div><span class="pill">Model C</span></div>
<div class="card"><h2>Upper Back IMU</h2>
<div class="axis">AX (g) <span id="u0">--</span></div><div class="axis">AY (g) <span id="u1">--</span></div><div class="axis">AZ (g) <span id="u2">--</span></div>
<div class="axis">GX (dps) <span id="u3">--</span></div><div class="axis">GY (dps) <span id="u4">--</span></div><div class="axis">GZ (dps) <span id="u5">--</span></div></div>
<div class="card"><h2>Left Shoulder IMU</h2>
<div class="axis">AX (g) <span id="l0">--</span></div><div class="axis">AY (g) <span id="l1">--</span></div><div class="axis">AZ (g) <span id="l2">--</span></div>
<div class="axis">GX (dps) <span id="l3">--</span></div><div class="axis">GY (dps) <span id="l4">--</span></div><div class="axis">GZ (dps) <span id="l5">--</span></div></div>
<div class="card"><h2>Right Shoulder IMU</h2>
<div class="axis">AX (g) <span id="r0">--</span></div><div class="axis">AY (g) <span id="r1">--</span></div><div class="axis">AZ (g) <span id="r2">--</span></div>
<div class="axis">GX (dps) <span id="r3">--</span></div><div class="axis">GY (dps) <span id="r4">--</span></div><div class="axis">GZ (dps) <span id="r5">--</span></div></div>
<div class="card"><h2>Flex Sensor 1</h2><div class="big" id="f1">--</div><div class="muted">ADC counts (0–4095)</div></div>
<div class="card"><h2>Flex Sensor 2</h2><div class="big" id="f2">--</div><div class="muted">ADC counts (0–4095)</div></div>
</div>
<p class="muted">ML outputs are experimental and trained on synthetic data; they are not a validated medical or ergonomic assessment.</p>
<h2>Latest ESP32 JSON</h2><pre id="raw">Waiting for data...</pre>
</main><script>
async function updateDashboard(){try{
 const response=await fetch('/api/latest');const data=await response.json();
 document.getElementById('status').textContent=data.status;
 document.getElementById('calibration').textContent=data.calibration_message||'';
 const p=data.prediction||{};
 document.getElementById('activity').textContent=p.activity||'Waiting';
 document.getElementById('posture').textContent=p.posture||'Waiting';
 document.getElementById('warning').textContent=p.warning||'Waiting';
 document.getElementById('probability').textContent='Poor-posture probability: '+(p.poor_probability==null?'--':(100*p.poor_probability).toFixed(1)+'%');
 document.getElementById('severity').textContent='Severity score: '+(p.severity_score==null?'--':p.severity_score);
 if(!data.reading)return;const d=data.reading;
 [['upper','u'],['left','l'],['right','r']].forEach(([s,prefix])=>{
   const v=d[s]||[];for(let i=0;i<6;i++)document.getElementById(prefix+i).textContent=(v[i]===undefined?'--':Number(v[i]).toFixed(3));
 });
 document.getElementById('f1').textContent=d.flex1??'--';document.getElementById('f2').textContent=d.flex2??'--';
 document.getElementById('raw').textContent=JSON.stringify(d,null,2);
 document.getElementById('updated').textContent='Last reading: '+new Date(data.received_at*1000).toLocaleTimeString()+' | samples: '+(p.sample_count??0);
}catch(e){document.getElementById('status').textContent='Cannot reach Flask';}}
updateDashboard();setInterval(updateDashboard,500);
</script></body></html>
"""


@app.route("/")
def dashboard():
    return render_template_string(HTML)


@app.route("/api/latest")
def latest():
    with lock:
        age = time.time() - last_received if last_received else None
        status = "No recent sensor data" if age is not None and age > 3 else serial_status
        return jsonify(
            status=status,
            reading=latest_reading,
            received_at=last_received,
            prediction=dict(latest_prediction),
            calibration_message=calibration_message,
        )


if __name__ == "__main__":
    print("Models loaded.")
    print(f"Activity features: {len(activity_features)}; posture features: {len(posture_features)}")
    print("Keep the Arduino Serial Monitor CLOSED while this dashboard reads the ESP32.")
    threading.Thread(target=serial_reader, daemon=True).start()
    app.run(host="127.0.0.1", port=5001, debug=False)
