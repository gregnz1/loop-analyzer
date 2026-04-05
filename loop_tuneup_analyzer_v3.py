#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from typing import Dict, Optional, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LOW = 3.9
VERY_LOW = 3.0
HIGH = 10.0
VERY_HIGH = 13.9


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def first_present(df: pd.DataFrame, candidates: List[str]):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def parse_dt(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def choose_time_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["Local Time", "Display Time", "Timestamp", "Timestamp (YYYY-MM-DDThh:mm:ss)", "time", "Date"]:
        if c in df.columns:
            return c
    return None


def time_filter(df: pd.DataFrame, time_col: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if time_col not in df.columns:
        return df.iloc[0:0].copy()
    s = parse_dt(df[time_col])
    out = df[(s >= start) & (s <= end)].copy()
    out[time_col] = parse_dt(out[time_col])
    return out


def load_clarity_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    time_col = first_present(df, ["Timestamp (YYYY-MM-DDThh:mm:ss)", "Timestamp", "Display Time"])
    glucose_col = first_present(df, ["Glucose Value (mmol/L)", "Glucose Value", "Value"])
    type_col = first_present(df, ["Event Type", "Type"])
    if time_col is None or glucose_col is None:
        raise ValueError("Could not find expected Clarity CSV columns")
    if type_col is not None:
        df = df[df[type_col].astype(str).str.contains("EGV|Glucose", case=False, na=False)].copy()
    df["time"] = pd.to_datetime(df[time_col], errors="coerce")
    df["glucose"] = pd.to_numeric(df[glucose_col], errors="coerce")
    return df.dropna(subset=["time", "glucose"])[["time", "glucose"]].sort_values("time").reset_index(drop=True)


def parse_food_carbs(nutrition_text):
    if pd.isna(nutrition_text):
        return np.nan
    try:
        data = nutrition_text if isinstance(nutrition_text, dict) else json.loads(nutrition_text)
        return float(data["carbohydrate"]["net"])
    except Exception:
        return np.nan


def load_tidepool_xlsx(path: Path) -> Dict[str, pd.DataFrame]:
    xl = pd.ExcelFile(path)
    wanted = [
        "Basal", "Bolus", "Device Event", "Food",
        "Basal Schedules", "BG Targets", "Carb Ratios", "Insulin Sensitivities"
    ]
    out: Dict[str, pd.DataFrame] = {}
    for sheet in wanted:
        if sheet in xl.sheet_names:
            df = pd.read_excel(path, sheet_name=sheet)
            tcol = choose_time_col(df)
            if tcol is not None:
                df[tcol] = parse_dt(df[tcol])
            out[sheet] = df
    if "Food" in out and "Nutrition" in out["Food"].columns:
        out["Food"]["carbs_g"] = out["Food"]["Nutrition"].apply(parse_food_carbs)
    return out


def glucose_metrics(cgm: pd.DataFrame) -> Dict[str, float]:
    g = cgm["glucose"].dropna()
    if len(g) == 0:
        return {}
    mean = float(g.mean())
    sd = float(g.std(ddof=1))
    return {
        "n": int(len(g)),
        "avg_glucose": mean,
        "sd": sd,
        "cv_pct": float(sd / mean * 100) if mean else np.nan,
        "tir_pct": float(((g >= LOW) & (g <= HIGH)).mean() * 100),
        "low_pct": float(((g < LOW) & (g >= VERY_LOW)).mean() * 100),
        "very_low_pct": float((g < VERY_LOW).mean() * 100),
        "high_pct": float(((g > HIGH) & (g <= VERY_HIGH)).mean() * 100),
        "very_high_pct": float((g > VERY_HIGH).mean() * 100),
        "min": float(g.min()),
        "max": float(g.max()),
    }


def hourly_glucose_metrics(cgm: pd.DataFrame) -> pd.DataFrame:
    df = cgm.copy()
    df["hour"] = df["time"].dt.hour
    rows = []
    for hour, grp in df.groupby("hour"):
        m = glucose_metrics(grp)
        m["hour"] = int(hour)
        rows.append(m)
    return pd.DataFrame(rows).sort_values("hour").reset_index(drop=True)


def date_glucose_metrics(cgm: pd.DataFrame) -> pd.DataFrame:
    df = cgm.copy()
    df["date"] = df["time"].dt.date
    rows = []
    for date, grp in df.groupby("date"):
        m = glucose_metrics(grp)
        m["date"] = pd.to_datetime(str(date))
        rows.append(m)
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def compare_named_windows(cgm: pd.DataFrame, windows: List[dict]) -> pd.DataFrame:
    rows = []
    for w in windows:
        grp = cgm[(cgm["time"] >= w["start"]) & (cgm["time"] <= w["end"])].copy()
        m = glucose_metrics(grp)
        m["window"] = w["label"]
        rows.append(m)
    return pd.DataFrame(rows)


def daily_insulin_summary(tide: Dict[str, pd.DataFrame], start, end) -> pd.DataFrame:
    frames = []
    if "Basal" in tide:
        b = time_filter(tide["Basal"], "Local Time", start, end).copy()
        if not b.empty:
            b["date"] = b["Local Time"].dt.date
            rate = pd.to_numeric(b.get("Rate"), errors="coerce")
            dur = pd.to_numeric(b.get("Duration (mins)"), errors="coerce")
            b["delivered"] = rate * (dur / 60.0)
            if "Payload" in b.columns:
                def payload_units(x):
                    try:
                        return json.loads(x).get("deliveredUnits", np.nan)
                    except Exception:
                        return np.nan
                b["payload_units"] = b["Payload"].apply(payload_units)
                b["delivered"] = b["payload_units"].fillna(b["delivered"])
            frames.append(b.groupby("date")["delivered"].sum().to_frame("basal_units"))
    if "Bolus" in tide:
        bo = time_filter(tide["Bolus"], "Local Time", start, end).copy()
        if not bo.empty:
            bo["date"] = bo["Local Time"].dt.date
            bo["units"] = pd.to_numeric(bo.get("Normal"), errors="coerce").fillna(0)
            subtype = bo.get("Sub Type")
            bo["is_automated"] = False if subtype is None else subtype.astype(str).str.contains("automated", case=False, na=False)
            frames.append(bo[bo["is_automated"]].groupby("date")["units"].sum().to_frame("auto_bolus_units"))
            frames.append(bo[~bo["is_automated"]].groupby("date")["units"].sum().to_frame("manual_bolus_units"))
    if "Food" in tide:
        f = time_filter(tide["Food"], "Local Time", start, end).copy()
        if not f.empty:
            f["date"] = f["Local Time"].dt.date
            if "carbs_g" not in f.columns:
                f["carbs_g"] = np.nan
            frames.append(f.groupby("date")["carbs_g"].sum().to_frame("carbs_g"))
            frames.append(f.groupby("date").size().to_frame("meal_entries"))
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, axis=1).fillna(0).reset_index()
    out["date"] = pd.to_datetime(out["date"].astype(str))
    out["total_insulin_units"] = out.get("basal_units", 0) + out.get("auto_bolus_units", 0) + out.get("manual_bolus_units", 0)
    return out.sort_values("date").reset_index(drop=True)


def settings_snapshot(tide: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    result = {}
    for sheet in ["Basal Schedules", "BG Targets", "Carb Ratios", "Insulin Sensitivities"]:
        if sheet in tide and not tide[sheet].empty and "Local Time" in tide[sheet].columns:
            df = tide[sheet].sort_values("Local Time").copy()
            latest = df["Local Time"].max()
            result[sheet] = df[df["Local Time"] == latest].copy()
    return result


def detect_settings_changes(tide: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    mapping = [
        ("Basal Schedules", "Basal Schedule Start", "Basal Schedule Rate"),
        ("BG Targets", "BG Target Start", "BG Low Target Setting"),
        ("Carb Ratios", "Carb Ratio Start", "Carb Ratio Amount"),
        ("Insulin Sensitivities", "Insulin Sensitivity Start", "Insulin Sensitivity Amount"),
    ]
    for sheet, start_col, value_col in mapping:
        if sheet not in tide or tide[sheet].empty:
            continue
        df = tide[sheet].copy()
        if not all(c in df.columns for c in ["Local Time", start_col, value_col]):
            continue
        df = df.sort_values(["Local Time", start_col])
        signatures = []
        for ts, g in df.groupby("Local Time", sort=True):
            sig = tuple(zip(g[start_col].astype(str), pd.to_numeric(g[value_col], errors="coerce").round(4)))
            signatures.append({"Local Time": ts, "signature": sig})
        sig = pd.DataFrame(signatures).sort_values("Local Time").reset_index(drop=True)
        sig["changed_vs_prev"] = sig["signature"].ne(sig["signature"].shift(1))
        sig["setting_sheet"] = sheet
        rows.append(sig[sig["changed_vs_prev"]][["setting_sheet", "Local Time", "signature"]])
    if not rows:
        return pd.DataFrame(columns=["setting_sheet", "Local Time", "signature"])
    return pd.concat(rows, ignore_index=True).sort_values("Local Time").reset_index(drop=True)


def infer_training_days(tide: Dict[str, pd.DataFrame], start, end, manual_override_dates=None) -> pd.DataFrame:
    dates = pd.date_range(start.normalize(), end.normalize(), freq="D")
    out = pd.DataFrame({"date": dates, "is_training_day": False, "training_signal": ""})
    if manual_override_dates:
        md = pd.DataFrame({"date": pd.to_datetime(manual_override_dates), "is_training_day": True, "training_signal": "manual_training_day"})
        out = out.merge(md, on="date", how="left", suffixes=("", "_manual"))
        out["is_training_day"] = out["is_training_day_manual"].fillna(False) | out["is_training_day"]
        out["training_signal"] = out["training_signal_manual"].fillna(out["training_signal"])
        return out[["date", "is_training_day", "training_signal"]]
    if "Device Event" in tide:
        de = time_filter(tide["Device Event"], "Local Time", start, end).copy()
        if not de.empty:
            text_cols = [c for c in ["Sub Type", "Reason"] if c in de.columns]
            if text_cols:
                de["signal_text"] = de[text_cols].astype(str).agg(" | ".join, axis=1)
                ex = de[de["signal_text"].str.contains("workout|exercise|activity", case=False, na=False)].copy()
                if not ex.empty:
                    ex["date"] = ex["Local Time"].dt.normalize()
                    s = ex.groupby("date")["signal_text"].apply(lambda x: "; ".join(sorted(set(x)))).reset_index(name="training_signal")
                    s["is_training_day"] = True
                    out = out.merge(s, on="date", how="left", suffixes=("", "_ev"))
                    out["is_training_day"] = out["is_training_day_ev"].fillna(False)
                    out["training_signal"] = out["training_signal_ev"].fillna("")
    return out[["date", "is_training_day", "training_signal"]]


def meal_analysis(cgm: pd.DataFrame, tide: Dict[str, pd.DataFrame], start, end) -> pd.DataFrame:
    if "Food" not in tide:
        return pd.DataFrame()
    f = time_filter(tide["Food"], "Local Time", start, end).copy()
    if f.empty:
        return pd.DataFrame()
    if "carbs_g" not in f.columns:
        f["carbs_g"] = np.nan
    bolus = tide.get("Bolus", pd.DataFrame()).copy()
    if not bolus.empty:
        bolus = time_filter(bolus, "Local Time", start, end)
        bolus["units"] = pd.to_numeric(bolus.get("Normal"), errors="coerce").fillna(0)
        subtype = bolus.get("Sub Type")
        bolus["is_automated"] = False if subtype is None else subtype.astype(str).str.contains("automated", case=False, na=False)
    rows = []
    for _, meal in f.iterrows():
        mt = meal["Local Time"]
        carbs = meal.get("carbs_g", np.nan)
        manual = pd.DataFrame()
        if not bolus.empty:
            manual = bolus[(~bolus["is_automated"]) & (bolus["Local Time"] >= mt - pd.Timedelta(minutes=30)) & (bolus["Local Time"] <= mt + pd.Timedelta(minutes=30))].copy()
        matched = not manual.empty
        prebolus_min = np.nan
        manual_units = 0.0
        if matched:
            manual["abs_delta_min"] = (manual["Local Time"] - mt).abs() / pd.Timedelta(minutes=1)
            best = manual.sort_values("abs_delta_min").iloc[0]
            prebolus_min = float((mt - best["Local Time"]) / pd.Timedelta(minutes=1))
            manual_units = float(manual["units"].sum())
        pre = cgm[(cgm["time"] >= mt - pd.Timedelta(minutes=15)) & (cgm["time"] <= mt + pd.Timedelta(minutes=15))]
        g0 = float(pre["glucose"].median()) if not pre.empty else np.nan
        peak_win = cgm[(cgm["time"] >= mt) & (cgm["time"] <= mt + pd.Timedelta(hours=3))]
        late_win = cgm[(cgm["time"] > mt + pd.Timedelta(hours=3)) & (cgm["time"] <= mt + pd.Timedelta(hours=5))]
        peak_glucose = float(peak_win["glucose"].max()) if not peak_win.empty else np.nan
        delta_peak = peak_glucose - g0 if pd.notna(peak_glucose) and pd.notna(g0) else np.nan
        low_3to5h = bool((late_win["glucose"] < LOW).any()) if not late_win.empty else False
        rows.append({
            "meal_time": mt,
            "date": mt.normalize(),
            "carbs_g": carbs,
            "matched_manual_bolus_30m": matched,
            "manual_bolus_units_30m_total": manual_units,
            "prebolus_minutes": prebolus_min,
            "glucose_at_meal": g0,
            "peak_glucose_0to3h": peak_glucose,
            "delta_peak_0to3h": delta_peak,
            "peak_over_10_0to3h": bool(pd.notna(peak_glucose) and peak_glucose > HIGH),
            "low_3to5h": low_3to5h,
        })
    return pd.DataFrame(rows).sort_values("meal_time").reset_index(drop=True)


def summarize_settings_text(settings: Dict[str, pd.DataFrame]) -> str:
    bits = []
    if "Basal Schedules" in settings:
        df = settings["Basal Schedules"].sort_values("Basal Schedule Start")
        bits.append("Basal: " + "; ".join([f"{t.strftime('%H:%M')} {r:g} U/hr" for t, r in zip(df["Basal Schedule Start"], df["Basal Schedule Rate"])]))
    if "BG Targets" in settings:
        df = settings["BG Targets"].sort_values("BG Target Start")
        bits.append("Targets: " + "; ".join([f"{t.strftime('%H:%M')} {low:g}-{high:g} {unit}" for t, low, high, unit in zip(df["BG Target Start"], df["BG Low Target Setting"], df["BG High Target Setting"], df["BG Units"])]))
    if "Carb Ratios" in settings:
        df = settings["Carb Ratios"].sort_values("Carb Ratio Start")
        bits.append("Carb ratio: " + "; ".join([f"{t.strftime('%H:%M')} 1:{amt:g} g" for t, amt in zip(df["Carb Ratio Start"], df["Carb Ratio Amount"])]))
    if "Insulin Sensitivities" in settings:
        df = settings["Insulin Sensitivities"].sort_values("Insulin Sensitivity Start")
        bits.append("ISF: " + "; ".join([f"{t.strftime('%H:%M')} {amt:g} mmol/L/U" for t, amt in zip(df["Insulin Sensitivity Start"], df["Insulin Sensitivity Amount"])]))
    return "\n".join([f"- {b}" for b in bits])


def decision_summary(recent_metrics: dict, prev_metrics: dict, meal_df: pd.DataFrame, hourly_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    def add(area, status, reason):
        rows.append({"area": area, "status": status, "reason": reason})
    if recent_metrics.get("avg_glucose", np.nan) > prev_metrics.get("avg_glucose", np.nan) + 0.2 and recent_metrics.get("low_pct", 0) >= prev_metrics.get("low_pct", 0):
        add("overall", "watch", "Recent window has higher average glucose and not fewer lows - variability increased rather than simple under-dosing.")
    else:
        add("overall", "leave", "No clear signal for a broad aggressive change across all settings.")
    if recent_metrics.get("low_pct", 0) + recent_metrics.get("very_low_pct", 0) >= 5.0:
        add("hypo exposure", "consider weakening", "Combined low + very low exposure is meaningfully elevated.")
    else:
        add("hypo exposure", "leave", "Hypo burden is not dominant enough to force a global pullback.")
    if not meal_df.empty:
        matched_pct = float(meal_df["matched_manual_bolus_30m"].mean() * 100)
        meal_big = meal_df[meal_df["carbs_g"] >= 25]
        peak_pct = float(meal_big["peak_over_10_0to3h"].mean() * 100) if not meal_big.empty else np.nan
        late_low_pct = float(meal_big["low_3to5h"].mean() * 100) if not meal_big.empty else np.nan
        if matched_pct < 80:
            add("meal timing", "consider strengthening habits", f"Only {matched_pct:.1f}% of meals had a manual bolus within 30 minutes.")
        else:
            add("meal timing", "leave", "Meal-to-bolus pairing is reasonably consistent.")
        if pd.notna(peak_pct) and pd.notna(late_low_pct) and peak_pct >= 20 and late_low_pct >= 15:
            add("meal cleanup pattern", "consider weakening late correction", "Meals show both post-meal peaks and later lows - suggests cleanup overshoot rather than simply weak meal dosing.")
        elif pd.notna(peak_pct) and peak_pct >= 25:
            add("meal cleanup pattern", "watch", "Meals frequently peak above range, but later lows are not dominant.")
        else:
            add("meal cleanup pattern", "leave", "Meal profile does not strongly suggest systematic cleanup overshoot.")
    if not hourly_df.empty:
        worst = hourly_df.sort_values(["low_pct", "very_low_pct"], ascending=False).iloc[0]
        if int(worst["hour"]) in [21, 22, 23, 0, 1, 2, 3]:
            add("late evening / overnight", "consider weakening", f"Worst hypo hour falls around {int(worst['hour']):02d}:00.")
        else:
            add("late evening / overnight", "watch", "Hypo windows are present, but not concentrated in the late evening / overnight block.")
    return pd.DataFrame(rows)


def plot_hourly_heatmap(hourly_df: pd.DataFrame, outpath: Path):
    if hourly_df.empty:
        return
    plot_df = hourly_df.set_index("hour")[["low_pct", "very_low_pct", "high_pct", "very_high_pct"]].T
    fig, ax = plt.subplots(figsize=(12, 2.8))
    im = ax.imshow(plot_df.values, aspect="auto")
    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h:02d}" for h in range(24)])
    ax.set_yticks(range(len(plot_df.index)))
    ax.set_yticklabels(plot_df.index)
    ax.set_title("Hourly low/high heatmap (%)")
    for i in range(plot_df.shape[0]):
        for j in range(plot_df.shape[1]):
            ax.text(j, i, f"{plot_df.values[i, j]:.1f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.03)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_meal_histogram(meal_df: pd.DataFrame, outpath: Path):
    if meal_df.empty:
        return
    vals = meal_df["delta_peak_0to3h"].dropna()
    if vals.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(vals, bins=20)
    ax.set_title("Meal peak rise within 3 h")
    ax.set_xlabel("Delta glucose from meal baseline (mmol/L)")
    ax.set_ylabel("Meal count")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_daily_trends(daily_df: pd.DataFrame, glucose_daily: pd.DataFrame, training_df: pd.DataFrame, outdir: Path):
    if daily_df.empty or glucose_daily.empty:
        return
    df = daily_df.merge(glucose_daily[["date", "avg_glucose", "tir_pct", "low_pct"]], on="date", how="outer").sort_values("date")
    if not training_df.empty:
        df = df.merge(training_df[["date", "is_training_day"]], on="date", how="left")
    else:
        df["is_training_day"] = False
    for col, title, fname in [
        ("avg_glucose", "Daily average glucose", "plot_daily_avg_glucose.png"),
        ("tir_pct", "Daily TIR (%)", "plot_daily_tir.png"),
        ("total_insulin_units", "Daily total insulin", "plot_daily_total_insulin.png"),
        ("carbs_g", "Daily carbs", "plot_daily_carbs.png"),
    ]:
        if col not in df.columns:
            continue
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df["date"], df[col], marker="o", linewidth=1)
        train = df[df["is_training_day"] == True]
        if not train.empty:
            ax.scatter(train["date"], train[col], marker="x")
        ax.set_title(title)
        ax.set_xlabel("Date")
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(outdir / fname, dpi=150)
        plt.close(fig)


def png_to_data_uri(path: Path) -> str:
    if not path.exists():
        return ""
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def write_html_dashboard(outdir: Path, windows_df: pd.DataFrame, decisions: pd.DataFrame, settings_text: str, notes_text: str):
    plot_files = [
        ("Hourly heatmap", "plot_hourly_heatmap.png"),
        ("Meal peak histogram", "plot_meal_peak_histogram.png"),
        ("Daily average glucose", "plot_daily_avg_glucose.png"),
        ("Daily TIR", "plot_daily_tir.png"),
        ("Daily total insulin", "plot_daily_total_insulin.png"),
        ("Daily carbs", "plot_daily_carbs.png"),
    ]
    cards = []
    for title, fname in plot_files:
        uri = png_to_data_uri(outdir / fname)
        if uri:
            cards.append(f'<section class="card"><h3>{title}</h3><img src="{uri}" alt="{title}"></section>')
    decision_items = "".join([f"<li><strong>{r.area}</strong> - {r.status}: {r.reason}</li>" for _, r in decisions.iterrows()])
    table_html = windows_df.to_html(index=False, border=0, classes="metrics")
    html = f'''<!doctype html>
<html><head><meta charset="utf-8"><title>Loop tune-up dashboard</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #222; }}
h1, h2, h3 {{ margin-bottom: 0.4rem; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; }}
.card {{ border: 1px solid #ddd; border-radius: 14px; padding: 14px; }}
img {{ width: 100%; height: auto; border-radius: 10px; }}
pre {{ white-space: pre-wrap; word-wrap: break-word; background: #fafafa; padding: 12px; border-radius: 10px; border: 1px solid #eee; }}
table.metrics {{ border-collapse: collapse; width: 100%; }}
table.metrics th, table.metrics td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: right; }}
table.metrics th:last-child, table.metrics td:last-child {{ text-align: left; }}
</style></head><body>
<h1>Loop tune-up dashboard</h1>
<section class="card"><h2>Window metrics</h2>{table_html}</section>
<section class="card"><h2>Decision summary</h2><ul>{decision_items}</ul></section>
<section class="card"><h2>Current settings snapshot</h2><pre>{settings_text}</pre></section>
<section class="card"><h2>Notes</h2><pre>{notes_text}</pre></section>
<div class="grid">{''.join(cards)}</div></body></html>'''
    (outdir / "dashboard.html").write_text(html, encoding="utf-8")


def markdown_report(outdir: Path, windows_df: pd.DataFrame, daily_df: pd.DataFrame, meal_df: pd.DataFrame, settings_text: str, settings_changes: pd.DataFrame, decisions: pd.DataFrame, training_df: pd.DataFrame, analysis_mode: str, notes_text: str) -> str:
    avg_daily = daily_df.mean(numeric_only=True).to_dict() if not daily_df.empty else {}
    meal_big = meal_df[meal_df["carbs_g"] >= 25].copy() if not meal_df.empty else pd.DataFrame()
    matched_pct = float(meal_df["matched_manual_bolus_30m"].mean() * 100) if not meal_df.empty else np.nan
    median_pre = float(meal_df.loc[meal_df["matched_manual_bolus_30m"], "prebolus_minutes"].median()) if not meal_df.empty and meal_df["matched_manual_bolus_30m"].any() else np.nan
    peak_pct = float(meal_big["peak_over_10_0to3h"].mean() * 100) if not meal_big.empty else np.nan
    late_low_pct = float(meal_big["low_3to5h"].mean() * 100) if not meal_big.empty else np.nan
    n_train = int(training_df["is_training_day"].sum()) if not training_df.empty else 0
    settings_change_text = "None detected."
    if not settings_changes.empty:
        settings_change_text = "\n".join([f"- {row['Local Time']}: {row['setting_sheet']}" for _, row in settings_changes.sort_values("Local Time").iterrows()])
    decision_md = "\n".join([f"- **{r.area}** - {r.status}: {r.reason}" for _, r in decisions.iterrows()]) if not decisions.empty else "- No automated decisions generated."
    report = f'''# Loop tune-up analysis v3

## Analysis mode
- {analysis_mode}

## Current settings snapshot
{settings_text}

## Settings changes detected in Tidepool
{settings_change_text}

## Window metrics
{windows_df.to_markdown(index=False)}

## Recent-window insulin and meal behaviour
- Basal delivered: {avg_daily.get('basal_units', np.nan):.1f} U/day
- Automated bolus: {avg_daily.get('auto_bolus_units', np.nan):.1f} U/day
- Manual bolus: {avg_daily.get('manual_bolus_units', np.nan):.1f} U/day
- Total insulin: {avg_daily.get('total_insulin_units', np.nan):.1f} U/day
- Carbs entered: {avg_daily.get('carbs_g', np.nan):.0f} g/day
- Meal entries: {avg_daily.get('meal_entries', np.nan):.1f}/day

## Meal-timing signals
- Meals with manual bolus within 30 min: {matched_pct:.1f}%
- Median prebolus time: {median_pre:.1f} min
- Meals >=25 g peaking >10 mmol/L within 3 h: {peak_pct:.1f}%
- Meals >=25 g followed by low at 3-5 h: {late_low_pct:.1f}%

## Training-day tagging
- Training days flagged in analysis window: {n_train}

## Decision summary
{decision_md}

## Notes
{notes_text}

## Output files
- report.md
- dashboard.html
- compare_windows.csv
- daily_insulin_recent.csv
- daily_glucose_metrics_recent.csv
- meal_analysis_recent.csv
- decision_summary.csv
- settings_changes.csv
- training_days_recent.csv
- plots (*.png)
'''
    (outdir / "report.md").write_text(report, encoding="utf-8")
    return report


def build_windows(args) -> List[dict]:
    if args.change_datetime:
        change_time = pd.Timestamp(args.change_datetime)
        return [
            {"label": f"pre_change_{args.pre_days}d", "start": change_time - pd.Timedelta(days=args.pre_days), "end": change_time - pd.Timedelta(seconds=1)},
            {"label": f"post_change_{args.post_days}d", "start": change_time, "end": change_time + pd.Timedelta(days=args.post_days) - pd.Timedelta(seconds=1)},
        ]
    recent_start = pd.Timestamp(args.recent_start)
    recent_end = pd.Timestamp(args.recent_end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    prev_start = pd.Timestamp(args.previous_start)
    prev_end = pd.Timestamp(args.previous_end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    return [
        {"label": "previous_30d", "start": prev_start, "end": prev_end},
        {"label": "recent_30d", "start": recent_start, "end": recent_end},
    ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clarity-csv", required=True, type=Path)
    p.add_argument("--tidepool-xlsx", required=True, type=Path)
    p.add_argument("--outdir", required=True, type=Path)
    p.add_argument("--recent-start")
    p.add_argument("--recent-end")
    p.add_argument("--previous-start")
    p.add_argument("--previous-end")
    p.add_argument("--change-datetime")
    p.add_argument("--pre-days", default=14, type=int)
    p.add_argument("--post-days", default=14, type=int)
    p.add_argument("--training-dates", nargs="*", default=[])
    p.add_argument("--notes-file", type=Path)
    args = p.parse_args()

    safe_mkdir(args.outdir)
    cgm = load_clarity_csv(args.clarity_csv)
    tide = load_tidepool_xlsx(args.tidepool_xlsx)
    windows = build_windows(args)
    analysis_mode = "settings_change_mode" if args.change_datetime else "recent_vs_previous_mode"

    windows_df = compare_named_windows(cgm, windows)
    windows_df.to_csv(args.outdir / "compare_windows.csv", index=False)

    recent_window = windows[-1]
    prev_window = windows[0]
    recent_start, recent_end = recent_window["start"], recent_window["end"]
    prev_start, prev_end = prev_window["start"], prev_window["end"]

    cgm_recent = cgm[(cgm["time"] >= recent_start) & (cgm["time"] <= recent_end)].copy()
    hourly_df = hourly_glucose_metrics(cgm_recent)
    hourly_df.to_csv(args.outdir / "hourly_glucose_metrics_recent.csv", index=False)

    glucose_daily = date_glucose_metrics(cgm_recent)
    glucose_daily.to_csv(args.outdir / "daily_glucose_metrics_recent.csv", index=False)

    daily_df = daily_insulin_summary(tide, recent_start, recent_end)
    daily_df.to_csv(args.outdir / "daily_insulin_recent.csv", index=False)

    meal_df = meal_analysis(cgm, tide, recent_start, recent_end)
    meal_df.to_csv(args.outdir / "meal_analysis_recent.csv", index=False)

    training_df = infer_training_days(tide, recent_start, recent_end, manual_override_dates=args.training_dates)
    training_df.to_csv(args.outdir / "training_days_recent.csv", index=False)

    settings = settings_snapshot(tide)
    for name, df in settings.items():
        df.to_csv(args.outdir / f"settings_snapshot_{name.lower().replace(' ', '_')}.csv", index=False)
    settings_text = summarize_settings_text(settings)

    settings_changes = detect_settings_changes(tide)
    settings_changes.to_csv(args.outdir / "settings_changes.csv", index=False)

    prev_metrics = glucose_metrics(cgm[(cgm["time"] >= prev_start) & (cgm["time"] <= prev_end)].copy())
    recent_metrics = glucose_metrics(cgm_recent)
    decisions = decision_summary(recent_metrics, prev_metrics, meal_df, hourly_df)
    decisions.to_csv(args.outdir / "decision_summary.csv", index=False)

    plot_hourly_heatmap(hourly_df, args.outdir / "plot_hourly_heatmap.png")
    plot_meal_histogram(meal_df, args.outdir / "plot_meal_peak_histogram.png")
    plot_daily_trends(daily_df, glucose_daily, training_df, args.outdir)

    notes_text = args.notes_file.read_text(encoding="utf-8") if args.notes_file and args.notes_file.exists() else "No external notes file supplied."
    report = markdown_report(args.outdir, windows_df, daily_df, meal_df, settings_text, settings_changes, decisions, training_df, analysis_mode, notes_text)
    write_html_dashboard(args.outdir, windows_df, decisions, settings_text, notes_text)
    print(report)


if __name__ == "__main__":
    main()
