"""
Spine-Grid demo dashboard with illustrative posture labels.

Reads the newest sensor_readings_*.csv created by sensor_logger.py.
It does NOT open the ESP32 serial port, so first run the logger to save a CSV,
then run this dashboard.

Run:
    python3 -m pip install flask pandas
    python3 spine_grid_demo_dashboard.py

Open:
    http://127.0.0.1:5002

IMPORTANT: The posture labels and confidence-style score are illustrative demo
outputs based on simple orientation heuristics. They are NOT predictions from
a validated model and do NOT establish real human posture.
"""
from pathlib import Path
import math
import io

import pandas as pd
from flask import Flask, request, render_template_string

BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__)

IMU_COLS = {
    "Upper back": ["upper_ax_g", "upper_ay_g", "upper_az_g"],
    "Left shoulder": ["left_ax_g", "left_ay_g", "left_az_g"],
    "Right shoulder": ["right_ax_g", "right_ay_g", "right_az_g"],
}
ALL_COLUMNS = [
    "upper_ax_g","upper_ay_g","upper_az_g","upper_gx_dps","upper_gy_dps","upper_gz_dps",
    "left_ax_g","left_ay_g","left_az_g","left_gx_dps","left_gy_dps","left_gz_dps",
    "right_ax_g","right_ay_g","right_az_g","right_gx_dps","right_gy_dps","right_gz_dps",
    "flex1","flex2",
]

def newest_csv():
    files = sorted((BASE_DIR / "data" / "raw_sensor_logs").glob("sensor_readings_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None

def load_frame(file_bytes=None):
    if file_bytes is not None:
        df = pd.read_csv(io.BytesIO(file_bytes))
        name = "Uploaded CSV"
    else:
        path = newest_csv()
        if not path:
            raise FileNotFoundError("No sensor_readings_*.csv found in data/raw_sensor_logs. Run sensor_logger.py first or upload a CSV.")
        df = pd.read_csv(path)
        name = path.name
    df.columns = [str(c).strip() for c in df.columns]
    for c in df.columns:
        if c in ALL_COLUMNS:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    required = ["upper_ax_g","upper_ay_g","upper_az_g"]
    if not all(c in df.columns for c in required):
        raise ValueError("CSV doesn't contain the expected logger columns such as upper_ax_g, upper_ay_g, upper_az_g.")
    df = df.dropna(subset=required).reset_index(drop=True)
    if df.empty:
        raise ValueError("No valid sensor rows found in the CSV.")
    return df, name

def mean_col(df, c):
    return float(df[c].mean()) if c in df.columns and df[c].notna().any() else None

def get_stats(df):
    stats = {}
    for c in ALL_COLUMNS:
        if c in df.columns and df[c].notna().any():
            s = df[c].dropna()
            stats[c] = {"mean": float(s.mean()), "min": float(s.min()), "max": float(s.max()), "range": float(s.max()-s.min()), "last": float(s.iloc[-1])}
    return stats

def classify_demo(df):
    """
    Simple transparent orientation heuristic, not a trained classifier.
    Uses mean upper-back accelerometer axes only. The labels describe the
    sensor's orientation pattern, not the person's real posture.
    """
    x, y, z = [mean_col(df, f"upper_a{axis}_g") for axis in "xyz"]
    mag = math.sqrt(x*x + y*y + z*z)
    # Heuristic buckets tuned only to give a readable demonstration using the
    # kinds of sensor orientations seen in this user's example recordings.
    if abs(z) >= 0.72:
        label = "Upright-like sensor orientation"
        detail = "Gravity is mostly along the upper sensor's Z axis."
        score = min(94, 72 + abs(z)*20)
        color = "blue"
    elif abs(y) >= 0.70:
        label = "Side-axis-dominant orientation"
        detail = "Gravity is mostly along the upper sensor's Y axis."
        score = min(92, 70 + abs(y)*20)
        color = "purple"
    elif abs(x) >= 0.72:
        label = "Large-tilt simulation"
        detail = "Gravity is mostly along the upper sensor's X axis."
        score = min(93, 70 + abs(x)*20)
        color = "orange"
    else:
        label = "Intermediate / mixed orientation"
        detail = "Gravity is spread across multiple axes."
        score = 76
        color = "teal"
    return {
        "label": label, "detail": detail, "score": round(score),
        "x": x, "y": y, "z": z, "magnitude": mag, "color": color
    }

def svg_chart(df, col, width=740, height=150):
    if col not in df.columns:
        return ""
    vals = pd.to_numeric(df[col], errors="coerce").dropna().tolist()
    if not vals:
        return ""
    if len(vals) > 150:
        stride = (len(vals)-1)/149
        vals = [vals[round(i*stride)] for i in range(150)]
    lo, hi = min(vals), max(vals)
    if math.isclose(lo, hi, abs_tol=1e-9):
        lo -= 0.5; hi += 0.5
    pad=10
    pts=[]
    for i,v in enumerate(vals):
        xx=pad+i*(width-2*pad)/max(1,len(vals)-1)
        yy=height-pad-(v-lo)*(height-2*pad)/(hi-lo)
        pts.append(f"{xx:.1f},{yy:.1f}")
    return (f'<svg viewBox="0 0 {width} {height}" role="img">'
            f'<line x1="{pad}" y1="{height/2}" x2="{width-pad}" y2="{height/2}" stroke="#e5eaf2"/>'
            f'<polyline points="{" ".join(pts)}" fill="none" stroke="#2867c7" stroke-width="2.4" stroke-linejoin="round" stroke-linecap="round"/>'
            f'<text x="{pad}" y="10" fill="#66758b" font-size="10">{hi:.3f}</text>'
            f'<text x="{pad}" y="{height-1}" fill="#66758b" font-size="10">{lo:.3f}</text></svg>')

def make_page(df, filename, note=""):
    stats = get_stats(df)
    pred = classify_demo(df)
    magnitude = {}
    for name, cols in IMU_COLS.items():
        if all(c in df.columns for c in cols):
            vals = (df[cols].pow(2).sum(axis=1, min_count=3)) ** .5
            magnitude[name] = float(vals.mean())
        else:
            magnitude[name] = None
    axis_cards = []
    groups = [
        ("Upper back", ["upper_ax_g","upper_ay_g","upper_az_g","upper_gx_dps","upper_gy_dps","upper_gz_dps"]),
        ("Left shoulder", ["left_ax_g","left_ay_g","left_az_g","left_gx_dps","left_gy_dps","left_gz_dps"]),
        ("Right shoulder", ["right_ax_g","right_ay_g","right_az_g","right_gx_dps","right_gy_dps","right_gz_dps"]),
    ]
    for title, cols in groups:
        cards=[]
        for c in cols:
            if c in stats:
                unit = "g" if "_a" in c and c.endswith("_g") else ("°/s" if "_g" in c and c.endswith("_dps") else "")
                cards.append({"name": c, "title": c.replace("_"," "), "unit": unit, **stats[c], "chart": svg_chart(df,c)})
        axis_cards.append({"title":title,"cards":cards})
    flex_cards=[]
    for c in ("flex1","flex2"):
        if c in stats:
            flex_cards.append({"name":c, "title":c, "unit":"ADC", **stats[c], "chart":svg_chart(df,c)})
    return render_template_string(TEMPLATE, filename=filename, note=note, rows=len(df), pred=pred,
                                  stats=stats, magnitudes=magnitude, groups=axis_cards, flex=flex_cards)

TEMPLATE = r"""
<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Spine-Grid Demo Dashboard</title>
<style>
:root{--ink:#17243a;--muted:#64748b;--line:#e3e9f2;--bg:#f3f6fb;--blue:#2867c7}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif}
main{max-width:1200px;margin:auto;padding:24px 16px 45px}h1{font-size:clamp(27px,4vw,36px);margin:7px 0}h2{font-size:20px;margin:0 0 13px}h3{font-size:15px;margin:0 0 5px}
.muted{color:var(--muted)}.small{font-size:12px}.tag{display:inline-block;background:#e5efff;color:#204e98;padding:5px 9px;border-radius:20px;font-weight:750;font-size:11px;letter-spacing:.04em}
.panel,.card,.metric{background:white;border:1px solid var(--line);border-radius:15px;padding:16px;box-shadow:0 3px 14px #1b33540a}
.header{display:flex;align-items:flex-start;justify-content:space-between;gap:14px;flex-wrap:wrap}.upload{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:17px 0}
input,button{padding:10px;border-radius:9px;border:1px solid var(--line);background:white}button{background:#205db4;color:white;border:0;font-weight:700;cursor:pointer}
.warning{margin:16px 0;background:#fff8e8;border:1px solid #f0d9a6;border-radius:12px;padding:13px}
.prediction{margin:17px 0;padding:22px;border-radius:18px;background:white;border:1px solid var(--line);border-top:5px solid #2867c7}
.prediction h2{font-size:25px;margin:7px 0}.score{font-size:29px;font-weight:800}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(285px,1fr));gap:13px}
.metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin:15px 0 23px}.metric .k{font-size:11px;color:var(--muted);text-transform:uppercase}.metric .v{font-size:25px;font-weight:800;margin:4px 0}
.section{margin:26px 0 12px}.card svg{width:100%;height:auto;background:#fbfcfe;border-radius:8px;margin:7px 0}
.stats{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:6px;margin-top:9px}.stat{background:#f5f7fb;border-radius:8px;padding:7px;min-width:0}.stat small{display:block;color:var(--muted);font-size:10px;text-transform:uppercase}.stat b{font-size:12px;overflow-wrap:anywhere}
table{border-collapse:collapse;width:100%;font-size:13px}td,th{text-align:right;padding:9px;border-bottom:1px solid var(--line);white-space:nowrap}td:first-child,th:first-child{text-align:left}.tablewrap{overflow:auto}
@media(max-width:500px){main{padding:16px 10px}.stats{grid-template-columns:repeat(2,1fr)}}
</style></head><body><main>
<div class="header"><div><span class="tag">SPINE-GRID · DEMO MODE</span><h1>Posture Sensor Dashboard</h1><p class="muted">Recorded IMU data with an illustrative orientation label.</p></div>
<div class="panel"><b>Recording</b><br>{{ filename }}<br><span class="muted small">{{ rows }} samples in this CSV</span></div></div>
{% if note %}<p class="muted small">{{ note }}</p>{% endif %}
<div class="warning"><b>Important for your presentation:</b> The label below is a simulated demo output from simple accelerometer-axis rules, not a validated posture-model prediction. The sensor boards in these recordings were on a table/being tilted, not attached to a person. Do not present the score as measured model accuracy.</div>
<section class="prediction">
 <span class="tag">SIMULATED ORIENTATION OUTPUT</span>
 <h2>{{ pred.label }}</h2>
 <p>{{ pred.detail }}</p>
 <p class="muted small">Rule-based demo indicator (not model confidence)</p>
 <div class="score">{{ pred.score }}<span style="font-size:15px;font-weight:500"> / 100 demo indicator</span></div>
 <div class="muted small">Upper IMU mean acceleration: X {{ "%.3f"|format(pred.x) }} g · Y {{ "%.3f"|format(pred.y) }} g · Z {{ "%.3f"|format(pred.z) }} g · magnitude {{ "%.3f"|format(pred.magnitude) }} g</div>
</section>
<div class="metrics">
{% for name, mag in magnitudes.items() %}
<div class="metric"><div class="k">{{ name }} · acceleration magnitude</div><div class="v">{{ "%.3f"|format(mag) if mag is not none else "N/A" }} g</div><div class="muted small">Mean magnitude over recording</div></div>
{% endfor %}
<div class="metric"><div class="k">Samples</div><div class="v">{{ rows }}</div><div class="muted small">From saved logger CSV</div></div>
</div>
<form class="upload" method="post" action="/upload" enctype="multipart/form-data"><label><b>Show another logger recording:</b></label><input type="file" name="csvfile" accept=".csv" required><button>Load CSV</button></form>
{% for group in groups %}
<div class="section"><h2>{{ group.title }}</h2><p class="muted small">Acceleration (g) and angular velocity (°/s)</p></div>
<div class="grid">{% for c in group.cards %}<article class="card"><h3>{{ c.name }}</h3>{{ c.chart|safe }}
<div class="stats"><div class="stat"><small>Mean</small><b>{{ "%.4f"|format(c.mean) }} {{ c.unit }}</b></div><div class="stat"><small>Min</small><b>{{ "%.4f"|format(c.min) }}</b></div><div class="stat"><small>Max</small><b>{{ "%.4f"|format(c.max) }}</b></div><div class="stat"><small>Range</small><b>{{ "%.4f"|format(c.range) }}</b></div></div></article>{% endfor %}</div>
{% endfor %}
{% if flex %}<div class="section"><h2>Flex sensors</h2></div><div class="grid">{% for c in flex %}<article class="card"><h3>{{ c.name }}</h3>{{ c.chart|safe }}<div class="stats"><div class="stat"><small>Mean</small><b>{{ "%.1f"|format(c.mean) }}</b></div><div class="stat"><small>Min</small><b>{{ "%.1f"|format(c.min) }}</b></div><div class="stat"><small>Max</small><b>{{ "%.1f"|format(c.max) }}</b></div><div class="stat"><small>Range</small><b>{{ "%.1f"|format(c.range) }}</b></div></div></article>{% endfor %}</div>{% endif %}
<div class="section"><h2>Channel summary</h2></div><div class="panel tablewrap"><table><thead><tr><th>Channel</th><th>Mean</th><th>Min</th><th>Max</th><th>Range</th><th>Unit</th></tr></thead><tbody>
{% for name,s in stats.items() %}<tr><td>{{ name }}</td><td>{{ "%.4f"|format(s.mean) }}</td><td>{{ "%.4f"|format(s.min) }}</td><td>{{ "%.4f"|format(s.max) }}</td><td>{{ "%.4f"|format(s.range) }}</td><td>{% if name in ["flex1","flex2"] %}ADC{% elif "_dps" in name %}°/s{% else %}g{% endif %}</td></tr>{% endfor %}
</tbody></table></div>
<p class="muted small" style="margin-top:22px">Spine-Grid demo visualization · labels are illustrative only; no clinical or real-world posture validation.</p>
</main></body></html>
"""

@app.route("/", methods=["GET"])
def index():
    try:
        df, name = load_frame()
        return make_page(df, name, "Automatically loaded the newest CSV in the project folder.")
    except Exception as e:
        return f"""<!doctype html><html><body style="font:16px Arial;max-width:760px;margin:40px auto;padding:20px">
        <h1>Spine-Grid Demo Dashboard</h1><p>{str(e)}</p>
        <form method="post" action="/upload" enctype="multipart/form-data">
        <input type="file" name="csvfile" accept=".csv" required><button>Upload CSV</button></form></body></html>""", 400

@app.route("/upload", methods=["POST"])
def upload():
    f = request.files.get("csvfile")
    if not f or not f.filename:
        return "Please choose a CSV.", 400
    try:
        df, name = load_frame(f.read())
        return make_page(df, name, "Loaded the selected saved recording.")
    except Exception as e:
        return f"<h1>Could not load CSV</h1><pre>{str(e)}</pre>", 400

if __name__ == "__main__":
    print("Spine-Grid demo dashboard (reads CSV only; no serial port access)")
    print("Open http://127.0.0.1:5002")
    app.run(host="127.0.0.1", port=5002, debug=False)
