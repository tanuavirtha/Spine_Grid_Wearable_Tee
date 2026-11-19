# Training report - Spine-Grid Tee posture models

Dataset: `spine_grid_posture_synthetic.csv`, 119,246 rows, data_source = synthetic. **Synthetic data: these results test the pipeline, not real-world accuracy.**

## 1. Feature reproducibility

Derived columns recomputed from raw data: max abs difference 0.0500 (rounding only); NaN-pattern mismatches: 0.

## 2. Windows and participant-independent split

2 s windows, 50 % overlap (hop 1 s), never crossing sessions: 4,746 windows, 165 features (mean, std, min, max, range of 33 channels).

| activity window | train | val | test |
| --- | --- | --- | --- |
| lying_down | 372 | 70 | 149 |
| running | 275 | 52 | 97 |
| sitting | 944 | 175 | 399 |
| standing | 581 | 113 | 245 |
| transition | 210 | 42 | 85 |
| walking | 585 | 121 | 231 |

Train = P01, P02, P03, P04, P05 | validation = P06 | test = P07, P08. Windows labelled `transition` (activity share < 60 % or transition rows) are excluded from Model A.

## 3. Model A - activity classification

Validation (P06) comparison:

| candidate | validation macro-F1 |
| --- | --- |
| RF depth=None leaf=1 | 0.976 |
| RF depth=None leaf=3 | 0.977 |
| RF depth=16 leaf=1 | 0.976 |
| RF depth=16 leaf=3 | 0.977 |
| HistGradientBoosting | 0.993 |
| ROCKET (time-series) | 0.917 |

Selected for deployment: **RF depth=None leaf=3** (Random Forest: small-data friendly, interpretable, exportable to the ESP32). Refit on train + validation, then scored once on the test participants.


**Model A (Random Forest) on test participants P07-P08** - accuracy 0.990, macro-F1 0.989

| class | precision | recall | f1-score | support |
| --- | --- | --- | --- | --- |
| sitting | 0.995 | 1.000 | 0.998 | 399.000 |
| standing | 1.000 | 0.955 | 0.977 | 245.000 |
| walking | 1.000 | 1.000 | 1.000 | 231.000 |
| running | 1.000 | 1.000 | 1.000 | 97.000 |
| lying_down | 0.943 | 1.000 | 0.971 | 149.000 |

Confusion matrix (rows = true, columns = predicted):

| true \ pred | sitting | standing | walking | running | lying_down |
| --- | --- | --- | --- | --- | --- |
| sitting | 399 | 0 | 0 | 0 | 0 |
| standing | 2 | 234 | 0 | 0 | 9 |
| walking | 0 | 0 | 231 | 0 | 0 |
| running | 0 | 0 | 0 | 97 | 0 |
| lying_down | 0 | 0 | 0 | 0 | 149 |

Conventional vs time-series model on the test participants:

| model | test accuracy | test macro-F1 |
| --- | --- | --- |
| Random Forest (selected) | 0.990 | 0.989 |
| HistGradientBoosting | 0.969 | 0.973 |
| ROCKET (time-series) | 0.978 | 0.983 |

Per-participant test accuracy: P07 0.989, P08 0.991

Top features: `accel_magnitude__range` (0.036), `imu_upper_ay__max` (0.036), `accel_magnitude__max` (0.035), `accel_magnitude__std` (0.034), `imu_upper_ay__mean` (0.033), `accel_magnitude__min` (0.030), `imu_upper_ay__min` (0.029), `imu_left_ay__min` (0.027), `imu_left_ay__max` (0.026), `imu_right_ay__mean` (0.025)

## 4. Model B - posture quality (good vs poor, activity-aware)

Decision threshold on P(poor) tuned on validation P06: **tau = 0.40** (F1 0.995). Critical-severity factor for Model C tuned on P06: **1.9x** the protocol criterion (warning-level macro-F1 0.767). Refit on train + validation.


**Model B on test, true activity context** - accuracy 0.970, macro-F1 0.966

| class | precision | recall | f1-score | support |
| --- | --- | --- | --- | --- |
| good | 0.980 | 0.976 | 0.978 | 491.000 |
| poor | 0.950 | 0.958 | 0.954 | 240.000 |

Confusion matrix (rows = true, columns = predicted):

| true \ pred | good | poor |
| --- | --- | --- |
| good | 479 | 12 |
| poor | 10 | 230 |

**Model B on test, activity context predicted by Model A (full pipeline)** - accuracy 0.970, macro-F1 0.966

| class | precision | recall | f1-score | support |
| --- | --- | --- | --- | --- |
| good | 0.980 | 0.976 | 0.978 | 491.000 |
| poor | 0.950 | 0.958 | 0.954 | 240.000 |

Confusion matrix (rows = true, columns = predicted):

| true \ pred | good | poor |
| --- | --- | --- |
| good | 479 | 12 |
| poor | 10 | 230 |

144 test windows are labelled `uncertain` (borderline magnitude or posture ramps); they are not scored. The model calls 82% of them poor.

Top features: `left_shoulder_pitch_deg__mean` (0.053), `imu_left_ay__mean` (0.051), `imu_right_az__mean` (0.048), `right_shoulder_pitch_deg__mean` (0.044), `imu_left_az__mean` (0.039), `imu_right_ay__mean` (0.037), `right_shoulder_roll_deg__mean` (0.032), `left_shoulder_pitch_deg__max` (0.030), `right_shoulder_roll_deg__max` (0.024), `shoulder_rounding_dev_deg__mean` (0.023)

## 5. Model C - warning logic vs. firmware rules (test participants)

Rule: alert when Model A says sitting/standing/walking AND P(poor) >= 0.40 for 5 s in a row (1 s decision steps). Level = critical if sustained >= 20 s or deviation severity >= 1.9x the protocol criterion, else advisory. Running, lying and transitions never alert.

| metric (test participants) | ML pipeline (A+B+C) | Firmware rules (unmodified) |
| --- | --- | --- |
| events | 20.00 | 20.00 |
| detected | 18.00 | 11.00 |
| missed | 2.00 | 9.00 |
| recall | 0.90 | 0.55 |
| alerts_on_short_poor_episodes | 5.00 | 2.00 |
| alerts_on_borderline_episodes | 9.00 | 8.00 |
| false_alerts_good_posture | 1.00 | 4.00 |
| false_alerts_per_hour | 2.97 | 11.89 |
| median_latency_s | 0.00 | 0.00 |
| p90_latency_s | 0.14 | 0.00 |

Test recording time: 20.2 min. An unsafe event counts as detected if an alert fires between poor-posture onset (5 s before the event is labelled unsafe) and 3 s after it ends; latency is measured from the unsafe onset (0 = the alert was already on when the posture became unsafe). Alert episodes that overlap no unsafe event are split into: real poor posture that ended before 5 s, borderline episodes the annotation marks `uncertain`, and false alerts during good posture.


**Warning level per 1 s decision step (ML pipeline)** - accuracy 0.889, macro-F1 0.637

| class | precision | recall | f1-score | support |
| --- | --- | --- | --- | --- |
| none | 0.992 | 0.918 | 0.953 | 1083.000 |
| advisory | 0.338 | 0.746 | 0.466 | 59.000 |
| critical | 0.459 | 0.531 | 0.493 | 64.000 |

Confusion matrix (rows = true, columns = predicted):

| true \ pred | none | advisory | critical |
| --- | --- | --- | --- |
| none | 994 | 57 | 32 |
| advisory | 7 | 44 | 8 |
| critical | 1 | 29 | 34 |

Saved models/activity_model.joblib, models/posture_model.joblib, models/warning_config.json. Run time 24 s.

