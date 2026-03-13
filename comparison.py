"""
pytu/comparison.py
==================
Validation of simulation results against published reference datasets:
  - HITU Simulator data (Zderic paper figures)
  - Zderic multi-duty-factor temperature data

Public API
----------
direct_hitu_comparison(sim_temp, z)
    Compare final temperature against HITU Simulator reference points.

compare_with_zderic_model(z, T, T_history, time_points, sonication_time, I0)
    Single-point comparison of this simulation against Zderic's curves.

compare_with_zderic_model_full(z, T_history, time_points, sonication_time)
    Multi-timepoint temperature progression comparison.

compare_with_zderic_model_full2(z, Q, kappa, density, c_p, blood_perfusion,
                                 dz_m, dt)
    Full multi-duty sweep comparison with Zderic's paper.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from . import config as cfg


# ---------------------------------------------------------------------------
# HITU Simulator comparison
# ---------------------------------------------------------------------------

def direct_hitu_comparison(sim_temp, z, save_csv="hitu_direct_comparison.csv",
                             save_png="hitu_direct_comparison.png"):
    """
    Compare simulated final temperature against HITU Simulator reference data.

    Parameters
    ----------
    sim_temp : ndarray  Final simulated temperature (°C) on grid z
    z : ndarray         Depth grid (cm)
    save_csv : str      Output CSV path
    save_png : str      Output PNG path

    Returns
    -------
    comparison_df : DataFrame
    """
    hitu_z    = cfg.HITU_Z
    hitu_temp = cfg.HITU_TEMP

    sim_interp_fn    = interp1d(z, sim_temp, bounds_error=False, fill_value="extrapolate")
    sim_at_hitu_pts  = sim_interp_fn(hitu_z)
    abs_diff         = np.abs(sim_at_hitu_pts - hitu_temp)
    rmse             = float(np.sqrt(np.mean((sim_at_hitu_pts - hitu_temp) ** 2)))
    max_diff         = float(np.max(abs_diff))

    comparison_df = pd.DataFrame({
        "Depth (cm)":         hitu_z,
        "HITU Temp (°C)":     hitu_temp,
        "Model Temp (°C)":    sim_at_hitu_pts,
        "Absolute Diff (°C)": abs_diff,
    })
    comparison_df.to_csv(save_csv, index=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(z, sim_temp, "r-", label="Enhanced Model", linewidth=2)
    ax.plot(hitu_z, hitu_temp, "bo--", label="HITU Simulator", linewidth=2, markersize=6)
    ax.axvspan(cfg.PANCREAS_START, cfg.PANCREAS_END, color="yellow", alpha=0.2,
               label="Pancreas Region")
    ax.set_xlabel("Depth (cm)", fontsize=12)
    ax.set_ylabel("Temperature (°C)", fontsize=12)
    ax.set_title(f"Direct Comparison with HITU Simulator\nRMSE: {rmse:.4f}°C", fontsize=14)
    ax.grid(True)
    ax.legend(fontsize=12)
    fig.savefig(save_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\nDirect Comparison with HITU Simulator:")
    print(f"  RMSE:              {rmse:.4f}°C")
    print(f"  Maximum difference: {max_diff:.4f}°C")

    return comparison_df


# ---------------------------------------------------------------------------
# Zderic single-point comparison
# ---------------------------------------------------------------------------

def compare_with_zderic_model(z, T, T_history, time_points, sonication_time,
                               I0=None,
                               save_png="zderic_model_comparison.png"):
    """
    Compare the current unfocused simulation (treated as 100% duty factor)
    against Zderic's temperature curves.

    Returns
    -------
    (max_pancreas_temp, max_overall_temp) : tuple of float
    """
    I0 = cfg.I0 if I0 is None else I0
    zderic = cfg.ZDERIC_DATA
    current_duration_min = sonication_time / 60.0

    pancreas_idx = np.where((z >= cfg.PANCREAS_START) & (z <= cfg.PANCREAS_END))
    your_max_pancreas = float(np.max(T[pancreas_idx]))
    your_max_overall  = float(np.max(T))

    print(f"\nYour simulation parameters:")
    print(f"  Duration: {current_duration_min:.2f} minutes")
    print(f"  Intensity: {I0} W/cm²")
    print(f"  Max pancreas temperature: {your_max_pancreas:.2f}°C")
    print(f"  Max overall temperature:  {your_max_overall:.2f}°C")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Pancreas temperature
    for (pct_key, color) in [("10pct", "b"), ("25pct", "g"),
                               ("50pct", "y"), ("100pct", "r")]:
        ax1.plot(zderic["duration_min"], zderic[f"pancreas_temp_{pct_key}"],
                 f"{color}-", label=f"OnShape {pct_key}", linewidth=2)
    ax1.plot(current_duration_min, your_max_pancreas, "ko", markersize=10,
             label="Your Simulation")
    ax1.set_xlabel("Sonication Duration (min)", fontsize=12)
    ax1.set_ylabel("Maximum Pancreas Temperature (°C)", fontsize=12)
    ax1.set_title("Pancreas Temperature: Your Simulation vs. Zderic Model", fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Overall temperature
    for (pct_key, color) in [("10pct", "b"), ("25pct", "g"),
                               ("50pct", "y"), ("100pct", "r")]:
        ax2.plot(zderic["duration_min"], zderic[f"overall_temp_{pct_key}"],
                 f"{color}-", label=f"OnShape {pct_key}", linewidth=2)
    ax2.plot(current_duration_min, your_max_overall, "ko", markersize=10,
             label="Your Simulation")
    ax2.set_xlabel("Sonication Duration (min)", fontsize=12)
    ax2.set_ylabel("Maximum Overall Temperature (°C)", fontsize=12)
    ax2.set_title("Overall Temperature: Your Simulation vs. Zderic Model", fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    fig.savefig(save_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return your_max_pancreas, your_max_overall


# ---------------------------------------------------------------------------
# Zderic full time-progression comparison
# ---------------------------------------------------------------------------

def compare_with_zderic_model_full(z, T_history, time_points, sonication_time,
                                    save_png="zderic_full_comparison.png",
                                    save_csv="zderic_full_comparison.csv"):
    """
    Extract temperatures at 1-minute intervals and overlay on Zderic's curves.

    Returns
    -------
    your_pancreas_temps, your_overall_temps : lists of float
    """
    zderic = cfg.ZDERIC_DATA
    minute_points = np.array([1, 2, 3, 4, 5]) * 60

    time_indices = []
    for sec in minute_points:
        if sec <= sonication_time:
            idx = np.abs(time_points - sec).argmin()
            time_indices.append(int(idx))
        else:
            print(f"  Warning: {sec / 60:.0f}-minute point exceeds simulation time.")

    pancreas_idx = np.where((z >= cfg.PANCREAS_START) & (z <= cfg.PANCREAS_END))
    your_pancreas_temps = []
    your_overall_temps  = []
    actual_minutes      = []

    for idx in time_indices:
        temp_slice = T_history[:, idx]
        your_pancreas_temps.append(float(np.max(temp_slice[pancreas_idx])))
        your_overall_temps.append(float(np.max(temp_slice)))
        actual_minutes.append(float(time_points[idx] / 60))

    print("\nYour simulation temperature progression:")
    print("Duration (min) | Pancreas Max (°C) | Overall Max (°C)")
    for i, m in enumerate(actual_minutes):
        print(f"{m:14.2f} | {your_pancreas_temps[i]:17.2f} | {your_overall_temps[i]:15.2f}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    for (pct_key, color) in [("10pct", "b"), ("25pct", "g"),
                               ("50pct", "y"), ("100pct", "r")]:
        ax1.plot(zderic["duration_min"], zderic[f"pancreas_temp_{pct_key}"],
                 f"{color}-", linewidth=2, label=f"OnShape {pct_key}")
        ax1.plot(zderic["duration_min"], zderic[f"obese_pancreas_temp_{pct_key}"],
                 f"{color}--", linewidth=1.5, alpha=0.5, label=f"OnShape Obese {pct_key}")
    ax1.plot(actual_minutes, your_pancreas_temps, "ko-", markersize=8, linewidth=2,
             label="Your Simulation")
    ax1.set_xlabel("Sonication Duration (min)", fontsize=12)
    ax1.set_ylabel("Maximum Pancreas Temperature (°C)", fontsize=12)
    ax1.set_title("Pancreas Temperature: Your Simulation vs. OnShape Model", fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8, ncol=2)

    for (pct_key, color) in [("10pct", "b"), ("25pct", "g"),
                               ("50pct", "y"), ("100pct", "r")]:
        ax2.plot(zderic["duration_min"], zderic[f"overall_temp_{pct_key}"],
                 f"{color}-", linewidth=2, label=f"OnShape {pct_key}")
        ax2.plot(zderic["duration_min"], zderic[f"obese_overall_temp_{pct_key}"],
                 f"{color}--", linewidth=1.5, alpha=0.5, label=f"OnShape Obese {pct_key}")
    ax2.plot(actual_minutes, your_overall_temps, "ko-", markersize=8, linewidth=2,
             label="Your Simulation")
    ax2.set_xlabel("Sonication Duration (min)", fontsize=12)
    ax2.set_ylabel("Maximum Overall Temperature (°C)", fontsize=12)
    ax2.set_title("Overall Temperature: Your Simulation vs. OnShape Model", fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8, ncol=2)

    fig.tight_layout()
    fig.savefig(save_png, dpi=900, bbox_inches="tight")
    plt.close(fig)

    comparison_df = pd.DataFrame({
        "Duration_min":      actual_minutes,
        "Your_Pancreas_Temp": your_pancreas_temps,
        "Your_Overall_Temp":  your_overall_temps,
    })
    n = len(actual_minutes)
    if n <= len(zderic["duration_min"]):
        for pct in ["10pct", "25pct", "50pct", "100pct"]:
            comparison_df[f"Zderic_Pancreas_{pct}"] = zderic[f"pancreas_temp_{pct}"][:n]
            comparison_df[f"Zderic_Overall_{pct}"]  = zderic[f"overall_temp_{pct}"][:n]

    comparison_df.to_csv(save_csv, index=False)
    return your_pancreas_temps, your_overall_temps


# ---------------------------------------------------------------------------
# Full multi-duty comparison with Zderic
# ---------------------------------------------------------------------------

def compare_with_zderic_model_full2(z, Q, kappa, density, c_p, blood_perfusion,
                                     dz_m, dt,
                                     save_png="duty_factor_comparison_onshape_python.png"):
    """
    Run all duty-factor simulations and produce a Zderic-style comparison plot.
    """
    from .pulsed import run_all_duty_simulations

    zderic    = cfg.ZDERIC_DATA
    duties    = [0.10, 0.25, 0.50, 0.75, 1.00]
    durations = [1, 2, 3, 4, 5]

    your_results = run_all_duty_simulations(
        z, Q, kappa, density, c_p, blood_perfusion, dz_m, dt,
        duties=duties, durations=durations
    )

    color_map = {"10": "r", "25": "g", "50": "b", "75": "m", "100": "k"}

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Zderic reference curves
    for (pct_key, color) in [("10pct", "r"), ("25pct", "g"),
                               ("50pct", "b"), ("100pct", "k")]:
        ax1.plot(zderic["duration_min"], zderic[f"pancreas_temp_{pct_key}"],
                 f"{color}--", label=f"OnShape {pct_key}", linewidth=2)
        ax2.plot(zderic["duration_min"], zderic[f"overall_temp_{pct_key}"],
                 f"{color}--", label=f"OnShape {pct_key}", linewidth=2)

    # Simulation curves — one series per duty factor
    plotted_duties = set()
    for key, data in your_results.items():
        duty_str = key.split("_")[0]
        color = color_map.get(duty_str, "grey")
        label = f"Python {duty_str}%" if duty_str not in plotted_duties else ""
        plotted_duties.add(duty_str)
        ax1.plot(data["time_points"], data["pancreas_temps"],
                 f"{color}-o", markersize=5, linewidth=1.5, label=label)
        ax2.plot(data["time_points"], data["overall_temps"],
                 f"{color}-o", markersize=5, linewidth=1.5, label=label)

    ax1.set_ylabel("Pancreas Temperature (°C)", fontsize=12)
    ax1.set_title("Duty Factor Comparison: Pancreas Temperature", fontsize=14)
    ax1.legend(fontsize=9, ncol=2)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Sonication Duration (min)", fontsize=12)
    ax2.set_ylabel("Overall Temperature (°C)", fontsize=12)
    ax2.set_title("Duty Factor Comparison: Overall Temperature", fontsize=14)
    ax2.legend(fontsize=9, ncol=2)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_png, dpi=900, bbox_inches="tight")
    plt.close(fig)

    print(f"\nDuty factor comparison saved to {save_png}")
