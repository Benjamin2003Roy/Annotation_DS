# ep06_classic_run.py
# EP-06 classic: Device reading = f(Assigned concentration)
# Configure measurand, target basis, and acceptance criteria IN CODE below.

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import OLSInfluence
from scipy import stats

# =========================
# ===== AC ================
# HB -> 5%
# RBC -> 5%
# WBC -> 7.5%
# HCT -> 5%
# PLT -> 12.5%
#==========================

# =========================
# ===== USER SETTINGS =====
# =========================
SELECT_MEASURAND = "PLT"      # <- pick any key present in DATASETS
TARGET_BASIS     = "Nominal"      # <- "MR" to use Mindray mean per level; "Nominal" to use label values
ACCEPTANCE_PCT   = 12.5    # <- ±% limit for response-domain bias checks

# =========================
# ===== LABELS/CONST  =====
# =========================
@dataclass(frozen=True)
class DeviceNames:
    mindray: str = "Mindray"
    pd4: str = "SigVet PD4"
    pd3: str = "SigVet PD3"

DEV = DeviceNames()

# =========================
# ======= DATASETS ========
# Each row: (level_label, mindray, pd4, pd3); exactly TWO rows per level_label
# Add more measurands under DATASETS as needed.
# =========================
DATASETS: Dict[str, List[Tuple[float, float, float, float]]] = {
    "HB": [
    (28.8, 28.8, 30.97, 30.02),
    (28.8, 29.1, 28.33, 27.27),
    (26.3, 26.3, 26.88, 26.12),
    (26.3, 26.1, 28.62, 27.58),
    (23.5, 23.5, 24.96, 24.29),
    (23.5, 23.4, 25.65, 24.88),
    (20.6, 20.6, 21.25, 20.63),
    (20.6, 20.5, 23.59, 22.64),
    (17.8, 17.8, 17.66, 17.21),
    (17.8, 17.8, 18.24, 17.57),
    (15.0, 15.0, 16.79, 16.18),
    (15.0, 15.0, 15.21, 14.48),
    (12.5, 12.5, 12.84, 12.46),
    (12.5, 12.6, 12.83, 12.14),
    (9.4, 9.4, 10.23, 9.6),
    (9.4, 9.4, 9.81, 9.0),
    (6.7, 6.7, 6.85, 6.27),
    (6.7, 6.8, 7.1, 6.63),
    (4.1, 4.1, 3.81, 3.18),
    (4.1, 4.1, 3.62, 3.01)
],
    "RBC": [
    (11.44, 11.44, 11.87, 11.6),
    (11.44, 11.49, 12.14, 12.0),
    (10.82, 10.82, 11.83, 11.68),
    (10.82, 10.49, 11.5, 11.33),
    (9.6, 9.6, 10.57, 10.48),
    (9.6, 9.64, 11.1, 10.93),
    (8.55, 8.55, 8.97, 8.82),
    (8.55, 8.43, 9.28, 9.15),
    (7.33, 7.33, 8.24, 8.07),
    (7.33, 7.42, 8.99, 9.1),
    (6.28, 6.28, 7.4, 7.29),
    (6.28, 6.28, 7.24, 7.17),
    (5.23, 5.23, 6.13, 6.08),
    (5.23, 5.16, 5.91, 5.85),
    (3.88, 3.88, 4.93, 4.84),
    (3.88, 3.85, 4.82, 4.71),
    (2.72, 2.72, 3.8, 3.66),
    (2.72, 2.83, 3.59, 3.51),
    (1.67, 1.67, 2.59, 2.56),
    (1.67, 1.67, 2.42, 2.42)
],
    "WBC": [
    (47.86, 47.86, 34.59, 35.19),
    (47.86, 48.22, 33.79, 35.59),
    (42.58, 42.58, 34.45, 34.65),
    (42.58, 43.01, 31.73, 32.11),
    (37.8, 37.8, 33.05, 32.34),
    (37.8, 38.24, 32.39, 33.17),
    (34.58, 34.58, 27.81, 27.33),
    (34.58, 33.34, 31.67, 29.79),
    (28.89, 28.89, 27.08, 26.96),
    (28.89, 28.76, 26.54, 27.09),
    (24.3, 24.3, 24.78, 25.4),
    (24.3, 24.47, 21.44, 22.12),
    (20.43, 20.43, 20.31, 19.92),
    (20.43, 20.46, 18.55, 18.44),
    (15.43, 15.43, 15.93, 16.12),
    (15.43, 15.14, 14.76, 14.87),
    (10.63, 10.63, 11.1, 10.96),
    (10.63, 10.75, 11.07, 11.07),
    (6.73, 6.73, 6.95, 6.12),
    (6.73, 6.46, 7.22, 6.53)
],
    "HCT": [
    (69.8, 69.8, 80.2, 79.04),
    (69.8, 70.1, 81.94, 81.59),
    (66.3, 66.3, 78.96, 78.77),
    (66.3, 64.4, 77.07, 76.58),
    (59.3, 59.3, 71.18, 71.4),
    (59.3, 59.8, 75.22, 74.89),
    (53.2, 53.2, 61.43, 60.5),
    (53.2, 52.3, 63.4, 63.7),
    (45.8, 45.8, 54.79, 54.45),
    (45.8, 46.3, 60.72, 62.28),
    (39.4, 39.4, 50.36, 49.68),
    (39.4, 39.5, 48.33, 49.11),
    (33.0, 33.0, 42.09, 42.03),
    (33.0, 32.7, 40.55, 40.49),
    (24.7, 24.7, 33.88, 33.38),
    (24.7, 24.5, 32.79, 32.72),
    (17.5, 17.5, 25.17, 24.39),
    (17.5, 18.3, 24.37, 24.27),
    (10.8, 10.8, 17.77, 17.64),
    (10.8, 10.8, 16.76, 16.78)
],
    "PLT": [
    (367, 367, 603.16, 582.59),
    (367, 381, 631.95, 605.76),
    (355, 355, 602.36, 511.37),
    (355, 343, 690.74, 601.73),
    (322, 322, 316.88, 531.21),
    (322, 300, 446.98, 523.22),
    (270, 270, 482.62, 291.6),
    (270, 264, 498.73, 409.72),
    (231, 231, 410.23, 399.76),
    (231, 241, 446.52, 416.85),
    (203, 203, 495.28, 462.23),
    (203, 194, 427.28, 442.88),
    (163, 163, 1112.66, 940.84),
    (163, 164, 452.84, 493.69),
    (133, 133, 548.86, 557.05),
    (133, 122, 226.85, 241.47),
    (82, 82, 129.78, 174.76),
    (82, 104, 187.54, 197.45),
    (63, 63, 165.4, 105.12),
    (63, 57, 612.35, 563.45)
]
}

# =========================
# ======== HELPERS ========
# =========================
def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s.strip())

def basis_tag() -> str:
    return "MR" if TARGET_BASIS.strip().lower() == "mr" else "Nominal"

def build_dataframe(measurand: str) -> pd.DataFrame:
    """
    Build tidy DataFrame with:
      - target: assigned concentration
          - MR basis: mean(Mindray) per level_label
          - Nominal basis: numeric level_label
      - replicate_id: 1/2 within level
      - device readings: mindray, pd4, pd3
      - level_label: original nominal label
    """
    if measurand not in DATASETS:
        raise KeyError(f"Unknown measurand '{measurand}'. Available: {list(DATASETS)}")

    raw = pd.DataFrame(DATASETS[measurand], columns=["level_label", "mindray", "pd4", "pd3"])
    raw["replicate_id"] = raw.groupby("level_label").cumcount() + 1

    counts = raw.groupby("level_label")["replicate_id"].nunique()
    bad = counts[counts != 2]
    if not bad.empty:
        raise ValueError(f"Each level must have exactly 2 rows. Problem levels: {list(bad.index)}")

    if basis_tag() == "MR":
        t = raw.groupby("level_label")["mindray"].mean().reset_index(name="target")
        df = raw.merge(t, on="level_label", how="left")
    else:
        df = raw.copy()
        df["target"] = df["level_label"].astype(float)

    df = df[["target", "replicate_id", "mindray", "pd4", "pd3", "level_label"]]
    df = df.sort_values(["target", "replicate_id"]).reset_index(drop=True)
    return df

def per_target_stats(df: pd.DataFrame, device_col: str) -> pd.DataFrame:
    g = df.groupby("target")[device_col]
    out = pd.DataFrame({"mean": g.mean(), "sd": g.std(ddof=1)})
    out["cv_%"] = (out["sd"] / out["mean"]).replace([np.inf, -np.inf], np.nan) * 100.0
    return out.reset_index().sort_values("target")

def _poly_design(x: np.ndarray, order: int) -> np.ndarray:
    cols = [np.ones_like(x)]
    for p in range(1, order + 1):
        cols.append(np.power(x, p))
    return np.column_stack(cols)

def fit_poly_and_table(x: np.ndarray, y: np.ndarray, order: int, alpha: float = 0.05) -> pd.DataFrame:
    """
    Fit y = β0 + β1 x + β2 x^2 + β3 x^3 (up to 'order').
    Return: β, SE, t, p, 95% CI, R^2, DF_resid, ±tcrit, Significance.
    """
    X = _poly_design(x, order)
    model = sm.OLS(y, X).fit()
    params, ses, tvals, pvals = model.params, model.bse, model.tvalues, model.pvalues
    ci = model.conf_int(alpha=alpha); r2 = model.rsquared; df_resid = int(model.df_resid)
    tcrit = stats.t.ppf(1 - alpha/2, df_resid)
    symbols = [f"β{i}" for i in range(order + 1)]
    sig = np.where(np.abs(tvals) > tcrit, "Significant", "Not Significant")
    return pd.DataFrame({
        "Coefficient": symbols,
        "Estimate": params, "SE": ses, "t_value": tvals, "p_value": pvals,
        "CI_lower": ci[:, 0], "CI_upper": ci[:, 1],
        "R_squared": [r2] * (order + 1), "DF_resid": [df_resid] * (order + 1),
        "Lower_significance_level": [-tcrit] * (order + 1),
        "Upper_significance_level": [tcrit] * (order + 1),
        "Significance": sig
    })

def regression_tables_for_device(df: pd.DataFrame, device_col: str) -> Dict[str, pd.DataFrame]:
    """
    EP-06 classic: X = target (assigned conc), Y = device mean reading.
    """
    st = per_target_stats(df, device_col)
    x = st["target"].to_numpy(float)  # independent: concentration
    y = st["mean"].to_numpy(float)    # dependent: device mean reading
    return {
        "Order 1": fit_poly_and_table(x, y, 1),
        "Order 2": fit_poly_and_table(x, y, 2),
        "Order 3": fit_poly_and_table(x, y, 3)
    }

def fit_poly_model(x: np.ndarray, y: np.ndarray, order: int):
    X = _poly_design(x, order)
    model = sm.OLS(y, X).fit()
    infl = OLSInfluence(model)
    return model, infl

def residuals_table_from_model(model, infl, x: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({
        "target": x,
        "Fitted_reading": model.fittedvalues,
        "Residual": model.resid,
        "Std_resid": infl.resid_studentized_internal,
        "Stud_resid_ext": infl.resid_studentized_external,
        "Leverage": infl.hat_matrix_diag,
        "Cooks_D": infl.cooks_distance[0],
        "DFFITS": infl.dffits[0],
    })

def residual_rse(model) -> float:
    return float(np.sqrt(model.ssr / model.df_resid))

def residuals_data_for_device(df: pd.DataFrame, device_col: str) -> dict:
    st = per_target_stats(df, device_col)
    x = st["target"].to_numpy(float)
    y = st["mean"].to_numpy(float)
    out = {}
    for order in (1, 2, 3):
        model, infl = fit_poly_model(x, y, order)
        rtab = residuals_table_from_model(model, infl, x, y)
        rse = residual_rse(model)
        out[order] = (model, infl, rtab, rse)
    return out

def ep06_significance_hint(tables: Dict[str, pd.DataFrame]) -> str:
    alpha = 0.05
    p3 = float(tables["Order 3"].loc[tables["Order 3"]["Coefficient"] == "β3", "p_value"].iloc[0])
    if p3 < alpha:
        return "Order 3: β3 significant ⇒ consider cubic."
    p2 = float(tables["Order 2"].loc[tables["Order 2"]["Coefficient"] == "β2", "p_value"].iloc[0])
    if p2 < alpha:
        return "Order 2: β2 significant ⇒ consider quadratic."
    return "Higher-order terms not significant ⇒ linear is adequate."

# ---------- Replicate/mean/diff table with TWO slopes ----------
def device_replicate_summary_table(df: pd.DataFrame, device_col: str) -> pd.DataFrame:
    """
    For each target concentration, list:
      1) Concentration
      2) Device replicate 1
      3) Device replicate 2
      4) Device mean
      5) |rep1 - rep2|
      6) (|rep1 - rep2| / mean) * 100   [%]
      7) (|rep1 - rep2|^2) / 2
      8) ((% diff)^2) / 2                 [%^2]
      9) Slope Rep1 vs previous = Δ(rep1)/Δ(target)
     10) Slope Rep2 vs previous = Δ(rep2)/Δ(target)
    """
    tmp = df[["target", "replicate_id", device_col]].copy()
    piv = tmp.pivot_table(index="target", columns="replicate_id", values=device_col, aggfunc="first")
    piv = piv.rename(columns={1: "rep1", 2: "rep2"}).reset_index()
    piv = piv.sort_values("target").reset_index(drop=True)

    piv["mean"] = (piv["rep1"] + piv["rep2"]) / 2.0
    piv["diff_abs"] = (piv["rep1"] - piv["rep2"]).abs()
    piv["diff_pct"] = (piv["diff_abs"] / piv["mean"]) * 100.0
    piv["diff2_over_2"] = (piv["diff_abs"] ** 2) / 2.0
    piv["diffpct2_over_2"] = (piv["diff_pct"] ** 2) / 2.0

    piv["slope_rep1_vs_prev"] = np.nan
    piv["slope_rep2_vs_prev"] = np.nan
    if len(piv) >= 2:
        t = piv["target"].to_numpy(float)
        r1 = piv["rep1"].to_numpy(float)
        r2 = piv["rep2"].to_numpy(float)
        dt = t[1:] - t[:-1]
        piv.loc[1:, "slope_rep1_vs_prev"] = (r1[1:] - r1[:-1]) / dt
        piv.loc[1:, "slope_rep2_vs_prev"] = (r2[1:] - r2[:-1]) / dt

    piv = piv.rename(columns={
        "target": "Concentration",
        "rep1": "Device replicate 1",
        "rep2": "Device replicate 2",
        "mean": "Device mean",
        "diff_abs": "Difference (abs)",
        "diff_pct": "Difference %",
        "diff2_over_2": "(Difference^2)/2",
        "diffpct2_over_2": "(Difference % ^2)/2",
        "slope_rep1_vs_prev": "Slope Rep1 vs previous",
        "slope_rep2_vs_prev": "Slope Rep2 vs previous",
    })[
        [
            "Concentration",
            "Device replicate 1",
            "Device replicate 2",
            "Device mean",
            "Difference (abs)",
            "Difference %",
            "(Difference^2)/2",
            "(Difference % ^2)/2",
            "Slope Rep1 vs previous",
            "Slope Rep2 vs previous",
        ]
    ]
    return piv

# ---------- EP-06 bias in response domain ----------
def actual_vs_predicted_device_reading(df: pd.DataFrame, device_col: str, order: int = 1):
    """
    Fit DeviceReading_mean = poly(target, order) on per-target means,
    then compare predicted reading vs ACTUAL mean reading at each target.
    """
    st = per_target_stats(df, device_col).sort_values("target")
    x = st["target"].to_numpy(float)
    y = st["mean"].to_numpy(float)
    X = _poly_design(x, order)
    model = sm.OLS(y, X).fit()
    yhat = model.fittedvalues
    out = pd.DataFrame({
        "target": x,
        "Actual_device_reading": y,
        "Predicted_device_reading": yhat,
    })
    out["Pct_diff_vs_actual"] = (out["Predicted_device_reading"] - out["Actual_device_reading"]) / out["Actual_device_reading"] * 100.0
    out["Within_±{:.0f}%".format(ACCEPTANCE_PCT)] = np.where(np.abs(out["Pct_diff_vs_actual"]) <= ACCEPTANCE_PCT, "Yes", "No")
    return out, model

def print_device_reading_bias(df_bias: pd.DataFrame, device_name: str):
    max_abs = float(np.max(np.abs(df_bias["Pct_diff_vs_actual"])))
    passed = max_abs <= ACCEPTANCE_PCT
    print(f"\n--- {device_name}: Predicted vs ACTUAL device reading ---")
    print(df_bias.to_string(index=False, float_format=lambda v: f"{v:0.5g}"))
    print(f"Max |% diff| = {max_abs:0.5g}%  → {'PASS' if passed else 'FAIL'} (limit ±{ACCEPTANCE_PCT}%)")

# -------- Plotting (X = concentration, Y = reading) --------
def _plot_one(ax, x_vals, y_vals, title=None, xlabel=None, ylabel=None, connect=True):
    if connect:
        ax.plot(x_vals, y_vals, "o-")
    else:
        ax.scatter(x_vals, y_vals, marker="o")
    if title: ax.set_title(title)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

def plot_all_devices_grid(df: pd.DataFrame, meas: str) -> str:
    """
    One PNG, 3x3:
      Rows = Mindray | pd4 | pd3
      Cols = Rep1 | Rep2 | Mean
    X = target (assigned concentration), Y = device reading.
    """
    devices = [("mindray", DEV.mindray), ("pd4", DEV.pd4), ("pd3", DEV.pd3)]
    rep1s, rep2s, means = {}, {}, {}
    for col, _ in devices:
        rep1 = df[df["replicate_id"] == 1][["target", col]].sort_values("target")
        rep2 = df[df["replicate_id"] == 2][["target", col]].sort_values("target")
        st   = per_target_stats(df, col)
        rep1s[col], rep2s[col], means[col] = rep1, rep2, st

    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(14, 12), sharex=False, sharey=False)
    col_titles = ["Replicate 1", "Replicate 2", "Mean of 2 Replicates"]
    for c, t in enumerate(col_titles): axes[0, c].set_title(t)

    for r, (col, dev_name) in enumerate(devices):
        ylab = f"{dev_name} reading"
        # Rep 1
        rep1 = rep1s[col]
        _plot_one(axes[r, 0], rep1["target"].values, rep1[col].values, xlabel=None, ylabel=ylab)
        # Rep 2
        rep2 = rep2s[col]
        _plot_one(axes[r, 1], rep2["target"].values, rep2[col].values, xlabel=None, ylabel=None)
        # Mean
        st = means[col]
        _plot_one(axes[r, 2], st["target"].values, st["mean"].values,
                  xlabel="Assigned concentration" if r == 2 else None, ylabel=None)

    title_target = "Mindray mean per level" if basis_tag() == "MR" else "Nominal level"
    fig.suptitle(f"{meas}: Device Reading vs Assigned Concentration ({title_target})", y=0.995)
    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    out_png = f"{meas}-{basis_tag()}-ep06classic_linearity_grid.png"
    plt.savefig(out_png, dpi=300, bbox_inches="tight"); plt.close()
    return out_png

def plot_residuals_grid_all_devices(df: pd.DataFrame, meas: str) -> str:
    """
    One PNG, 3x3:
      Rows = Mindray | pd4 | pd3
      Cols = Order 1 | 2 | 3
    Residual vs fitted READING; title shows Sy.x.
    """
    devices = [("mindray", DEV.mindray), ("pd4", DEV.pd4), ("pd3", DEV.pd3)]
    res = {col: residuals_data_for_device(df, col) for col, _ in devices}
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(14, 12), sharex=False, sharey=False)
    for r, (col, dev_name) in enumerate(devices):
        for c, order in enumerate((1, 2, 3)):
            model, infl, rtab, rse = res[col][order]
            ax = axes[r, c]
            ax.scatter(rtab["Fitted_reading"], rtab["Residual"])
            ax.axhline(0.0, linestyle="--")
            ax.set_xlabel("Fitted reading" if r == 2 else "")
            ax.set_ylabel(f"{dev_name}\nResidual" if c == 0 else "")
            ax.grid(True, alpha=0.3)
            ax.set_title(f"Order {order} — Sy.x={rse:0.5g}")
    fig.suptitle(f"{meas}: Residuals vs Fitted Reading — Orders 1–3 by Device", y=0.995)
    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    out_png = f"{meas}-{basis_tag()}-ep06classic_residuals_grid.png"
    plt.savefig(out_png, dpi=300, bbox_inches="tight"); plt.close()
    return out_png

# =========================
# ========= MAIN ==========
# =========================
def capture_output_to_file(outfile: str, func):
    with open(outfile, "w", encoding="utf-8") as f, redirect_stdout(f):
        func()
    print(f"\n✅ All results have been written to {outfile}")

def run(meas: str):
    df = build_dataframe(meas)

    tgt_name = "Mindray mean per level" if basis_tag() == "MR" else "Nominal level"
    print(f"\nTidy data (target = {tgt_name}; all rows):")
    print(df.to_string(index=False))

    # Replicate/mean/diff tables
    print("\n=== Replicates / Mean / Differences by device (X = Concentration, Y = Reading) ===")
    for col, name in [("mindray", DEV.mindray), ("pd4", DEV.pd4), ("pd3", DEV.pd3)]:
        tbl = device_replicate_summary_table(df, col)
        print(f"\n--- {name} ---")
        print(tbl.to_string(index=False, float_format=lambda v: f"{v:0.5g}"))

    # Regressions: DeviceReading = f(Concentration)
    print("\n=== Polynomial regression on per-target MEANS: DeviceReading = f(Assigned Concentration) ===")
    for col, name in [("mindray", DEV.mindray), ("pd4", DEV.pd4), ("pd3", DEV.pd3)]:
        tables = regression_tables_for_device(df, col)
        print(f"\n--- {name} ---")
        for label, tbl in tables.items():
            print(f"\n{label}")
            print(tbl.to_string(index=False, float_format=lambda v: f"{v:0.5g}"))
        print("→", ep06_significance_hint(tables))

    # Residual Standard Error summary
    print("\n=== Residual Standard Error (Sy.x) summary (DeviceReading models) ===")
    rows = []
    for col, name in [("mindray", DEV.mindray), ("pd4", DEV.pd4), ("pd3", DEV.pd3)]:
        res = residuals_data_for_device(df, col)
        rows.append({
            "Device": name,
            "Order 1 Sy.x": f"{res[1][3]:0.5g}",
            "Order 2 Sy.x": f"{res[2][3]:0.5g}",
            "Order 3 Sy.x": f"{res[3][3]:0.5g}",
        })
    print(pd.DataFrame(rows).to_string(index=False))

    # EP-06 bias (response domain)
    print("\n=== Predicted vs ACTUAL device reading (bias vs actual reading, EP-06) ===")
    # choose polynomial order per earlier t-tests; defaults are safe
    order_map = {"mindray": 1, "pd4": 1, "pd3": 2}
    for col, name in [("mindray", DEV.mindray), ("pd4", DEV.pd4), ("pd3", DEV.pd3)]:
        df_bias, _ = actual_vs_predicted_device_reading(df, col, order=order_map[col])
        print_device_reading_bias(df_bias, name)

    # === Repeatability summary at end (EP-06 two-replicate) ===
    print("\n=== Repeatability (EP-06 two-replicate): SDr & CVr vs acceptance ===")
    rep_rows = []
    for col, name in [("mindray", DEV.mindray), ("pd4", DEV.pd4), ("pd3", DEV.pd3)]:
        sdr, L, _ = sdr_cvr_from_two_reps(df, col, as_percent=False)
        cvr_pct, _, _ = sdr_cvr_from_two_reps(df, col, as_percent=True)
        verdict = "PASS" if np.abs(cvr_pct) <= ACCEPTANCE_PCT else "FAIL"
        rep_rows.append({
            "Device": name,
            "Levels": L,
            "SDr": f"{sdr:0.5g}",
            "CVr%": f"{cvr_pct:0.5g}",
            "Limit%": f"{ACCEPTANCE_PCT:0.5g}",
            "Result": verdict,
        })
    rep_df = pd.DataFrame(rep_rows, columns=["Device","Levels","SDr","CVr%","Limit%","Result"])
    print(rep_df.to_string(index=False))

    # Plots
    line_png = plot_all_devices_grid(df, meas)
    print(f"\n✅ Saved line plots grid PNG: {line_png}")
    res_png = plot_residuals_grid_all_devices(df, meas)
    print(f"✅ Saved residuals grid PNG: {res_png}")


def sdr_cvr_from_two_reps(df: pd.DataFrame, device_col: str, as_percent: bool = False):
    """
    EP-06 repeatability for exactly two replicates per level (L levels).
      d_i = y_i1 - y_i2
      SDr = sqrt( sum(d_i^2) / (2*L) )
    If as_percent=True, compute d_i% = 100*(y1 - y2)/mean_i (same units across levels),
      CVr = sqrt( sum(d_i% ^2) / (2*L) )   --> returns percent.
    Returns: value (SDr or CVr%), L, and a per-level diagnostics DataFrame.
    """
    tmp = df[["target", "replicate_id", device_col]].copy()
    piv = tmp.pivot(index="target", columns="replicate_id", values=device_col)
    piv = piv.rename(columns={1: "rep1", 2: "rep2"}).sort_index()
    L = len(piv)

    if not as_percent:
        d = (piv["rep1"] - piv["rep2"]).to_numpy(float)
        sdr = float(np.sqrt(np.sum(d**2) / (2.0 * L)))
        per_level = pd.DataFrame({
            "target": piv.index.astype(float),
            "rep1": piv["rep1"].to_numpy(float),
            "rep2": piv["rep2"].to_numpy(float),
            "diff_abs": np.abs(d),
            "(diff^2)/2": (d**2) / 2.0,
        })
        return sdr, L, per_level
    else:
        mean_i = piv[["rep1", "rep2"]].mean(axis=1).replace(0, np.nan)
        d_pct = 100.0 * (piv["rep1"] - piv["rep2"]) / mean_i
        cvr = float(np.sqrt(np.nansum(d_pct.to_numpy(float)**2) / (2.0 * L)))
        per_level = pd.DataFrame({
            "target": piv.index.astype(float),
            "rep1": piv["rep1"].to_numpy(float),
            "rep2": piv["rep2"].to_numpy(float),
            "mean_level": mean_i.to_numpy(float),
            "diff_%": np.abs(d_pct.to_numpy(float)),
            "(diff_%^2)/2": (d_pct.to_numpy(float)**2) / 2.0,
        })
        return cvr, L, per_level

def main():
    out_txt = f"{SELECT_MEASURAND}-{basis_tag()}-ep06classic_results.txt"
    capture_output_to_file(out_txt, lambda: run(SELECT_MEASURAND))

if __name__ == "__main__":
    main()