"""Offline analysis of the released Spine-Grid dataset (paper Section 4.4, Figs. 10-18).

Usage:  python analysis/offline_analysis.py [--no-figs]

Reads data/spine_grid_complete_sensor_dataset_15-10_16-10.csv and writes
analysis/results/summary.json, analysis/results/replay_windows.csv and figures.

Method (paper Section 3.5): each session is split into a start-up burst (stale packets
queued in the serial buffer; inter-sample gap < 10 ms) and a steady 25 Hz stream.
The first 1 s (25 samples) of the steady stream is the baseline; the rest is cut into
non-overlapping 0.5 s windows (12 samples). Accelerometers are smoothed with a 5-sample
centred moving average, tilt = max over the three IMUs of the angle to the baseline
gravity vector, D = |f1-f1_0| + |f2-f2_0| with invalid flex2 samples contributing 0.
A window is classified from its mean tilt / mean D with the hardware thresholds.
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data" / "spine_grid_complete_sensor_dataset_15-10_16-10.csv"
OUT = Path(__file__).resolve().parent / "results"
IMUS = ("upper", "left", "right")
BURST_GAP_S, BASE_N, WIN_N, SMOOTH = 0.010, 25, 12, 5
WARN_DEG, CRIT_DEG, WARN_ADC, CRIT_ADC = 8.0, 16.0, 80.0, 120.0


def split_burst(g):
    gap = np.diff(g.elapsed_s.values) > BURST_GAP_S
    k = int(np.argmax(gap)) + 1 if gap.any() else 0
    return g.iloc[:k], g.iloc[k:].reset_index(drop=True)


def deviations(s, base_n=BASE_N):
    th = []
    for n in IMUS:
        a = s[[f"{n}_ax_g", f"{n}_ay_g", f"{n}_az_g"]].rolling(SMOOTH, center=True, min_periods=1).mean().values
        a0 = a[:base_n].mean(0)
        c = (a @ a0) / (np.linalg.norm(a, axis=1) * np.linalg.norm(a0))
        th.append(np.degrees(np.arccos(np.clip(c, -1, 1))))
    tilt = np.max(th, axis=0)
    f1 = s.flex1.values.astype(float); f2 = s.flex2.values.astype(float)
    inv = s.flex2_invalid.values.astype(bool)
    f10 = f1[:base_n].mean()
    ok = ~inv[:base_n]
    f20 = f2[:base_n][ok].mean() if ok.any() else np.nan
    D = np.nan_to_num(np.abs(f1 - f10) + np.where(inv, 0.0, np.abs(f2 - f20)))
    return tilt, D


def classify(t, dflex, wd=WARN_DEG, cd=CRIT_DEG, wa=WARN_ADC, ca=CRIT_ADC):
    return "critical" if (t >= cd or dflex >= ca) else "warning" if (t >= wd or dflex >= wa) else "normal"


def main(figs=True):
    OUT.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(CSV)
    sessions = sorted(d.session_id.unique())
    sid = {s: f"S{i+1}" for i, s in enumerate(sessions)}
    res, rows, bursts, steady_dt = {}, [], [], []
    res["samples"] = len(d); res["sessions"] = len(sessions)
    res["samples_by_date"] = d.recording_date.value_counts().to_dict()
    res["missing_values"] = int(d.drop(columns=["notes"]).isna().sum().sum())  # "notes" is free text, empty by design
    res["flex2_invalid_samples"] = int(d.flex2_invalid.sum())
    res["flex2_invalid_pct"] = round(100 * d.flex2_invalid.mean(), 1)
    res["flex2_invalid_by_session"] = {sid[k]: int(v) for k, v in d.groupby("session_id").flex2_invalid.sum().items() if v}
    res["flex1_invalid_samples"] = int(d.flex1_invalid.sum())
    all_t, all_D = [], []
    for s in sessions:
        g = d[d.session_id == s].sort_values("sample_index")
        b, st = split_burst(g)
        bursts.append(int((g.elapsed_s <= 0.05).sum())); steady_dt += list(np.diff(st.elapsed_s.values) * 1000)
        tilt, D = deviations(st)
        n = (len(st) - BASE_N) // WIN_N
        for i in range(n):
            sl = slice(BASE_N + i * WIN_N, BASE_N + (i + 1) * WIN_N)
            t, dd = float(tilt[sl].mean()), float(D[sl].mean())
            rows.append(dict(session=sid[s], window=i, tilt_deg=round(t, 3), flex_dev_adc=round(dd, 2), level=classify(t, dd)))
        all_t.append(tilt[BASE_N:]); all_D.append(D[BASE_N:])
    w = pd.DataFrame(rows); w.to_csv(OUT / "replay_windows.csv", index=False)
    res["startup_burst_packets"] = dict(min=min(bursts), max=max(bursts), mean=round(float(np.mean(bursts)), 1))
    res["steady_interval_ms"] = dict(median=round(float(np.median(steady_dt)), 1), mean=round(float(np.mean(steady_dt)), 1), sd=round(float(np.std(steady_dt)), 1))
    vc = w.level.value_counts()
    res["replay_windows"] = dict(total=len(w), normal=int(vc.get("normal", 0)), warning=int(vc.get("warning", 0)), critical=int(vc.get("critical", 0)))
    res["replay_by_session"] = {k: v.to_dict() for k, v in w.groupby("session")["level"].value_counts().unstack(fill_value=0).iterrows()}
    s2 = d[d.session_id == sessions[1]]; s19 = d[d.session_id == sessions[-1]]
    res["flex_response_S2_to_S19"] = dict(
        flex1=[round(s2.flex1.mean(), 1), round(s19.flex1.mean(), 1)],
        flex2=[round(s2[~s2.flex2_invalid].flex2.mean(), 1), round(s19[~s19.flex2_invalid].flex2.mean(), 1)])
    T, Dall = w.tilt_deg.values, w.flex_dev_adc.values  # per 0.5 s window
    res["threshold_sensitivity_pct_of_windows"] = {
        "tilt>=8deg": round(100 * (T >= 8).mean(), 1), "tilt>=16deg": round(100 * (T >= 16).mean(), 1),
        "flex>=80adc": round(100 * (Dall >= 80).mean(), 1), "flex>=120adc": round(100 * (Dall >= 120).mean(), 1)}
    (OUT / "summary.json").write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps(res, indent=2, default=str))
    if figs:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 3)); ax.hist(np.clip(steady_dt, 0, 120), bins=60, log=True)
        ax.set(xlabel="steady-stream interval (ms)", ylabel="count (log)", title="Fig. 11 steady-stream intervals"); fig.tight_layout(); fig.savefig(OUT / "fig11_intervals.png", dpi=200)
        tab = w.groupby("session")["level"].value_counts().unstack(fill_value=0).reindex(columns=["normal", "warning", "critical"], fill_value=0)
        tab = tab.reindex(sorted(tab.index, key=lambda x: int(x[1:])))
        fig, ax = plt.subplots(figsize=(8, 3)); tab.plot(kind="bar", stacked=True, ax=ax, color=["#4a9", "#e9a820", "#c33"])
        ax.set(ylabel="0.5 s windows", title="Fig. 16 offline replay of the sensor rule"); fig.tight_layout(); fig.savefig(OUT / "fig16_replay.png", dpi=200)
        xs = np.linspace(0, 30, 121); xf = np.linspace(0, 250, 126)
        fig, a = plt.subplots(1, 2, figsize=(8, 3))
        a[0].plot(xs, [100 * (T >= x).mean() for x in xs]); a[0].axvline(WARN_DEG, ls="--", c="#e9a820"); a[0].axvline(CRIT_DEG, ls="--", c="#c33"); a[0].set(xlabel="tilt threshold (deg)", ylabel="% windows flagged")
        a[1].plot(xf, [100 * (Dall >= x).mean() for x in xf]); a[1].axvline(WARN_ADC, ls="--", c="#e9a820"); a[1].axvline(CRIT_ADC, ls="--", c="#c33"); a[1].set(xlabel="flex threshold (ADC)")
        fig.suptitle("Fig. 18 threshold sensitivity (per 0.5 s window)"); fig.tight_layout(); fig.savefig(OUT / "fig18_sensitivity.png", dpi=200)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--no-figs", action="store_true")
    main(figs=not ap.parse_args().no_figs)
