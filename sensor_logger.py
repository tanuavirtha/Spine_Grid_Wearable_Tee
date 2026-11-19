import json
import time
import csv
import serial
from serial.tools import list_ports
from datetime import datetime

BAUD = 115200
DURATION = 5

ports = list(list_ports.comports())
port = next(
    (p.device for p in ports
     if any(x in (p.device + " " + (p.description or "")).lower()
            for x in ["usbmodem", "usbserial", "esp32", "cp210", "ch340"])),
    None
)

if not port:
    print("ESP32 port not found. Available ports:")
    for p in ports:
        print(p.device, p.description)
    raise SystemExit(1)

print("Using:", port)
print("Keep all sensors connected and still on the table.")
input("Press ENTER to collect readings for 5 seconds...")

rows = []
with serial.Serial(port, BAUD, timeout=1) as ser:
    time.sleep(2)
    ser.reset_input_buffer()
    start = time.monotonic()

    while time.monotonic() - start < DURATION:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        try:
            d = json.loads(line)
            if not all(k in d for k in ["upper", "left", "right", "flex1", "flex2"]):
                continue

            row = {"elapsed_s": round(time.monotonic() - start, 3)}
            for sensor in ["upper", "left", "right"]:
                values = d[sensor]
                if len(values) < 6:
                    continue
                for i, axis in enumerate(["ax_g", "ay_g", "az_g", "gx_dps", "gy_dps", "gz_dps"]):
                    row[f"{sensor}_{axis}"] = float(values[i])
            row["flex1"] = float(d["flex1"])
            row["flex2"] = float(d["flex2"])
            rows.append(row)
        except (json.JSONDecodeError, TypeError, ValueError, KeyError):
            continue

if len(rows) < 2:
    print("Not enough readings collected:", len(rows))
    raise SystemExit(1)

import pathlib
OUT_DIR = pathlib.Path(__file__).resolve().parent / "data" / "raw_sensor_logs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
filename = OUT_DIR / ("sensor_readings_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv")
with open(filename, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"\nCollected {len(rows)} readings in approximately {DURATION} seconds.")
print("Saved:", filename)
print("\nMIN / MAX / MEAN")
print(f"{'Sensor axis':<22} {'MIN':>12} {'MAX':>12} {'MEAN':>12}")
print("-" * 60)

for key in rows[0]:
    if key == "elapsed_s":
        continue
    vals = [r[key] for r in rows if key in r]
    print(f"{key:<22} {min(vals):>12.4f} {max(vals):>12.4f} {sum(vals)/len(vals):>12.4f}")
