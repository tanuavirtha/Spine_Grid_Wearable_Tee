# Spine-Grid Tee — code and data

Code, data and analysis for the paper **"Spine-Grid Tee: A Design-Integrated Multi-Sensor Garment for Continuous Upper-Body Posture Monitoring with Real-Time Alerting"**.

The prototype: ESP32-S3 + MPU6050 IMUs (upper back, left/right shoulder) + two flex sensors → JSON over USB serial → Python processing (calibration, features, activity/posture models, transparent sensor rule) → Flask dashboard with a SPINE-style node & message monitor.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Live dashboard (ESP32 connected, Arduino Serial Monitor closed)
python serial_flask_dashboard_spine.py          # open http://127.0.0.1:5001

# 2. Reproduce the offline analysis of the released dataset (paper Section 4.4, Figs. 10-18)
python analysis/offline_analysis.py             # writes analysis/results/
```

## Repository layout

| Path | Purpose |
|---|---|
| `serial_flask_dashboard_spine.py` | **Canonical live dashboard** used for the paper (Figs. 7-9): validation, calibration, activity + posture models, sensor rule (Eq. 3), persistence alerts, SPINE-style node/message monitor |
| `posture_features.py` | Shared feature/angle definitions |
| `sensor_logger.py` | Records 5 s of live readings to `data/raw_sensor_logs/` |
| `spine_grid_csv_replay_demo.py` | Illustrative dashboard that replays a logged CSV (no serial port) |
| `data/spine_grid_complete_sensor_dataset_15-10_16-10.csv` | **Released real dataset**: 3,227 samples, 19 sessions, 32 columns |
| `data/raw_sensor_logs/` | The 19 original logger CSVs the dataset was curated from |
| `analysis/offline_analysis.py` | Offline analysis behind Section 4.4 / Figs. 10-18 |
| `generate_dataset.py`, `train_models.py`, `test_inference.py`, `test_e_validation.py`, `models/`, `reports/` | Synthetic-data generator and model training/testing pipeline |
| `legacy_dashboards/` | Earlier dashboard iterations, kept for history only (superseded) |
| `firmware/`, `wokwi/` | ESP32-S3 sketch and Wokwi project (see the READMEs there) |

## Alert thresholds (paper Eq. 3, hardware values)

| Level | IMU tilt | Combined flex deviation |
|---|---|---|
| Warning | ≥ 8° | ≥ 80 ADC |
| Critical | ≥ 16° | ≥ 120 ADC |

A critical deviation sustained for 5 s raises a critical alert. Constants are at the top of `serial_flask_dashboard_spine.py`. (The Wokwi simulation uses 400/900 ADC because potentiometers stand in for the flex strips.) These are prototype thresholds from bench observations, not clinically validated limits.

## Released dataset

`data/spine_grid_complete_sensor_dataset_15-10_16-10.csv`: 19 sessions of about 5 s at 25 Hz (after a start-up burst of ~45 buffered packets per session), recorded on a bench prototype **without a human wearer**. Columns: session metadata (`recording_date`, `session_id`, `source_file`, `session_label`, `label_confidence`, `notes`), timing (`sample_index`, `elapsed_s`), 18 IMU channels (`{upper,left,right}_{ax,ay,az}_g`, `_{gx,gy,gz}_dps`), `flex1`, `flex2` (12-bit ADC), and quality columns `flex1_invalid`, `flex2_invalid`, `sensor_quality_flag`. 395 samples (12.2%) have flex2 dropped to 0–4 ADC (loose connection). Only two sessions carry a label; the dataset supports hardware characterisation only.

Note: the original logger files are all named 2026-10-16; `recording_date` in the curated CSV is the date assigned in the paper's 15/16 October split (see `recording_date_note`).

## Models

The activity and posture models in `models/` were trained on the **synthetic** table produced by `generate_dataset.py` (seed 20261006; 119,246 rows, 8 participants) and are experimental: they have not been validated on readings from this prototype, and `reports/` results test the pipeline, not real-world accuracy. To retrain: `python generate_dataset.py && python train_models.py`.

## Limitations

SPINE integration is a Python adapter demonstration over USB serial; the Java SPINE coordinator is not connected. The bench prototype has three IMUs (the lumbar IMU is part of the design but not yet fitted).
