"""
pytu/thermal.py
===============
Pennes bioheat finite-difference thermal solver.

Public API
----------
compute_heating_term(att_dB, I, dz)
    Returns the volumetric heat source Q (W/m³) and alpha_Np (Np/m).

compute_thermal_diffusivity(k_thermal, density, c_p)
    Returns kappa = k / (rho * c_p)  in m²/s.

compute_stable_dt(kappa, dz, dt_requested)
    Applies the explicit FD stability criterion and returns the safe dt.

run_thermal_simulation(...)
    Main bioheat loop; returns T, T_history, time_points.

analyze_pancreas_heating(z, T, label)
    Computes and prints pancreas region temperature statistics.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm

from . import config as cfg


# ---------------------------------------------------------------------------
# Pre-processing helpers
# ---------------------------------------------------------------------------

def compute_heating_term(att_dB, I, dz=None):
    """
    Compute the volumetric heat source term Q at each depth point.

    Q = 2 * alpha_Np * I   [W/m³]
    (I converted from W/cm² to W/m² and alpha from dB/cm to Np/m)

    Parameters
    ----------
    att_dB : ndarray   Attenuation (dB/cm) at each depth
    I : ndarray        Intensity (W/cm²) at each depth
    dz : float, optional  Spatial step (cm)  — not used in formula but kept
                          for API consistency.

    Returns
    -------
    Q : ndarray        Volumetric heat source (W/m³)
    alpha_Np : ndarray Attenuation in Np/m
    """
    alpha_Np = att_dB * (np.log(10) / 20) * 100   # dB/cm → Np/m
    Q = 2 * alpha_Np * I * 10_000                  # I: W/cm² → W/m²
    return Q, alpha_Np


def compute_thermal_diffusivity(k_thermal, density, c_p):
    """
    Thermal diffusivity  kappa = k / (rho * c_p)  [m²/s].

    Zero-density points are returned as zero to avoid division by zero.
    """
    kappa = np.where(density > 0, k_thermal / (density * c_p), 0.0)
    return kappa


def compute_stable_dt(kappa, dz_m, dt_requested=None):
    """
    Compute the largest time step satisfying the explicit FD stability
    criterion:  dt <= 0.5 * dz² / kappa_max

    Parameters
    ----------
    kappa : ndarray   Thermal diffusivity (m²/s) — may contain zeros
    dz_m : float      Spatial step in metres
    dt_requested : float, optional
        Upper bound on dt (s). Defaults to config.DT_DEFAULT.

    Returns
    -------
    dt : float   Safe time step (s)
    """
    dt_requested = cfg.DT_DEFAULT if dt_requested is None else dt_requested
    kappa_pos = kappa[kappa > 0]
    if len(kappa_pos) == 0:
        return dt_requested
    dt_max = 0.5 * dz_m ** 2 / np.min(kappa_pos)
    return min(dt_requested, dt_max)


# ---------------------------------------------------------------------------
# Main thermal solver
# ---------------------------------------------------------------------------

def run_thermal_simulation(z, Q, kappa, density, c_p, blood_perfusion,
                            sonication_time=None, dt=None, dz_m=None,
                            T_init=37.0):
    """
    Run the Pennes bioheat equation forward in time using an explicit
    finite-difference scheme.

    The loop computes, at each interior grid point:
        dT/dt = kappa * d²T/dz² + perfusion_term + source_term

    Adiabatic (zero-flux) boundary conditions are applied at both ends.

    Parameters
    ----------
    z : ndarray             Depth grid (cm)
    Q : ndarray             Volumetric heat source (W/m³)
    kappa : ndarray         Thermal diffusivity (m²/s)
    density : ndarray       Density (kg/m³)
    c_p : ndarray           Specific heat (J/(kg·K))
    blood_perfusion : ndarray  Blood perfusion (ml_blood/(ml_tissue·s))
    sonication_time : float, optional  Total sonication duration (s)
    dt : float, optional    Time step (s); if None, stability criterion is used
    dz_m : float, optional  Spatial step in metres; derived from config DZ if None
    T_init : float          Initial uniform temperature (°C). Default 37.

    Returns
    -------
    T : ndarray             Final temperature distribution (°C)
    T_history : ndarray     Shape (len(z), nt) full temperature history
    time_points : ndarray   Time axis (s) matching T_history columns
    """
    sonication_time = cfg.SONICATION_TIME if sonication_time is None else sonication_time
    dz_m = cfg.DZ / 100 if dz_m is None else dz_m

    if dt is None:
        dt = compute_stable_dt(kappa, dz_m)

    nt = int(sonication_time / dt)
    time_points = np.linspace(0, sonication_time, nt)

    T = np.ones_like(z) * T_init
    T_history = np.zeros((len(z), nt))
    T_history[:, 0] = T

    blood_den = cfg.BLOOD_DENSITY
    blood_cp  = cfg.BLOOD_SPECIFIC_HEAT
    blood_T   = cfg.BLOOD_TEMP

    print("Running thermal simulation (Pennes bioheat model)...")
    for n in tqdm(range(1, nt)):
        dT = np.zeros_like(T)
        for i in range(1, len(z) - 1):
            diffusion  = kappa[i] * (T[i + 1] - 2 * T[i] + T[i - 1]) / dz_m ** 2
            perfusion  = (blood_perfusion[i] * blood_den * blood_cp
                          * (blood_T - T[i]) / density[i] / c_p[i])
            source     = Q[i] / (density[i] * c_p[i])
            dT[i]      = dt * (diffusion + perfusion + source)

        # Adiabatic boundary conditions
        dT[0]  = dT[1]
        dT[-1] = dT[-2]

        T = T + dT
        T_history[:, n] = T

    return T, T_history, time_points


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze_pancreas_heating(z, T, label=""):
    """
    Compute and print temperature statistics within the pancreas depth range.

    Parameters
    ----------
    z : ndarray   Depth grid (cm)
    T : ndarray   Final temperature (°C)
    label : str   Optional label for printed output (e.g. "Focused Beam")

    Returns
    -------
    metrics : dict
        Keys: min_temp, max_temp, mean_temp, std_temp, max_position
    """
    pancreas_idx = np.where((z >= cfg.PANCREAS_START) & (z <= cfg.PANCREAS_END))
    pancreas_temps = T[pancreas_idx]

    min_temp   = float(np.min(pancreas_temps))
    max_temp   = float(np.max(pancreas_temps))
    mean_temp  = float(np.mean(pancreas_temps))
    std_temp   = float(np.std(pancreas_temps))
    max_pos    = float(z[pancreas_idx][np.argmax(pancreas_temps)])

    header = f"Pancreas Thermal Analysis" + (f" ({label})" if label else "")
    print(f"\n{header}:")
    print(f"  Minimum temperature: {min_temp:.2f}°C")
    print(f"  Maximum temperature: {max_temp:.2f}°C")
    print(f"  Mean temperature:    {mean_temp:.2f}°C")
    print(f"  Standard deviation:  {std_temp:.2f}°C")
    print(f"  Position of maximum: {max_pos:.2f} cm")

    return {
        "min_temp":     min_temp,
        "max_temp":     max_temp,
        "mean_temp":    mean_temp,
        "std_temp":     std_temp,
        "max_position": max_pos,
    }


def save_pancreas_analysis_csv(metrics, path="pancreas_thermal_analysis.csv"):
    """Save pancreas thermal analysis dict to CSV."""
    pd.DataFrame({
        "Metric": ["Minimum", "Maximum", "Mean", "Std Dev", "Max Position"],
        "Value":  [
            f"{metrics['min_temp']:.2f}°C",
            f"{metrics['max_temp']:.2f}°C",
            f"{metrics['mean_temp']:.2f}°C",
            f"{metrics['std_temp']:.2f}°C",
            f"{metrics['max_position']:.2f} cm",
        ],
    }).to_csv(path, index=False)
