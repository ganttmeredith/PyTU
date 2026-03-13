"""
pytu/pulsed.py
==============
Duty-cycle and pulsed waveform thermal simulations.

The legacy script contained two separate implementations of
`run_pulsed_simulation`.  This module consolidates them into one
correct implementation that uses an explicit ON/OFF duty cycle
(current_time % pulse_period < on_time).

Public API
----------
run_pulsed_simulation(z, Q, kappa, density, c_p, blood_perfusion,
                       dz_m, dt, duty_factor, duration_min)
    Runs a single duty-factor simulation; returns per-minute temperature lists.

run_all_duty_simulations(z, Q, kappa, density, c_p, blood_perfusion,
                          dz_m, dt, duties, durations)
    Runs all requested duty-factor / duration combinations.
"""

import numpy as np
from tqdm import tqdm

from . import config as cfg


def run_pulsed_simulation(z, Q, kappa, density, c_p, blood_perfusion,
                           dz_m, dt, duty_factor, duration_min):
    """
    Simulate ultrasound heating with an explicit ON/OFF duty cycle.

    During the ON phase Q is applied at full strength; during the OFF phase
    only thermal diffusion and blood perfusion act on the tissue.

    Parameters
    ----------
    z : ndarray              Depth grid (cm)
    Q : ndarray              Full-power volumetric heat source (W/m³)
    kappa : ndarray          Thermal diffusivity (m²/s)
    density : ndarray        Density (kg/m³)
    c_p : ndarray            Specific heat (J/(kg·K))
    blood_perfusion : ndarray Blood perfusion (ml_blood/(ml_tissue·s))
    dz_m : float             Spatial step (m)
    dt : float               Time step (s)
    duty_factor : float      Duty cycle fraction [0, 1]
    duration_min : float     Total simulation duration (minutes)

    Returns
    -------
    time_points : list[int]     Minute indices at which temperatures are sampled
    pancreas_temps : list[float] Max pancreas temperature at each minute
    overall_temps : list[float]  Max overall temperature at each minute
    """
    print(f"\nRunning {duty_factor * 100:.0f}% duty factor simulation "
          f"for {duration_min} min...")

    duration_sec = duration_min * 60
    nt = int(duration_sec / dt)

    T_pulsed = np.ones_like(z) * 37.0
    T_history_pulsed = np.zeros((len(z), nt))
    T_history_pulsed[:, 0] = T_pulsed

    pulse_period = 1.0          # 1-second duty cycle period
    on_time = pulse_period * duty_factor

    blood_den = cfg.BLOOD_DENSITY
    blood_cp  = cfg.BLOOD_SPECIFIC_HEAT
    blood_T   = cfg.BLOOD_TEMP

    for n in tqdm(range(1, nt)):
        current_time = n * dt
        Q_scaled = Q if (current_time % pulse_period) < on_time else np.zeros_like(Q)

        dT = np.zeros_like(T_pulsed)
        for i in range(1, len(z) - 1):
            diffusion = kappa[i] * (T_pulsed[i + 1] - 2 * T_pulsed[i] + T_pulsed[i - 1]) / dz_m ** 2
            perfusion = (blood_perfusion[i] * blood_den * blood_cp
                         * (blood_T - T_pulsed[i]) / density[i] / c_p[i])
            source    = Q_scaled[i] / (density[i] * c_p[i])
            dT[i]     = dt * (diffusion + perfusion + source)

        dT[0]  = dT[1]
        dT[-1] = dT[-2]
        T_pulsed += dT
        T_history_pulsed[:, n] = T_pulsed

    # Sample temperatures at each whole-minute mark
    pancreas_idx = np.where((z >= cfg.PANCREAS_START) & (z <= cfg.PANCREAS_END))
    time_points   = []
    pancreas_temps = []
    overall_temps  = []

    for minute in range(1, int(duration_min) + 1):
        idx = int(minute * 60 / dt)
        if idx < nt:
            time_points.append(minute)
            pancreas_temps.append(float(np.max(T_history_pulsed[pancreas_idx, idx])))
            overall_temps.append(float(np.max(T_history_pulsed[:, idx])))

    return time_points, pancreas_temps, overall_temps


def run_all_duty_simulations(z, Q, kappa, density, c_p, blood_perfusion,
                              dz_m, dt,
                              duties=None, durations=None):
    """
    Run pulsed simulations for every combination of duty factor and duration.

    Parameters
    ----------
    duties : list of float, optional
        Duty-cycle fractions. Default: [0.10, 0.25, 0.50, 0.75, 1.00]
    durations : list of int, optional
        Durations in minutes. Default: [1, 2, 3, 4, 5]

    Returns
    -------
    results : dict
        Keys are f"{int(duty*100)}_{duration}min"; values are dicts with
        keys: time_points, pancreas_temps, overall_temps.
    """
    duties    = [0.10, 0.25, 0.50, 0.75, 1.00] if duties    is None else duties
    durations = [1, 2, 3, 4, 5]                 if durations is None else durations

    results = {}
    for duty in duties:
        for duration in durations:
            key = f"{int(duty * 100)}_{duration}min"
            tp, pt, ot = run_pulsed_simulation(
                z, Q, kappa, density, c_p, blood_perfusion,
                dz_m, dt, duty, duration
            )
            results[key] = {
                "time_points":   tp,
                "pancreas_temps": pt,
                "overall_temps":  ot,
            }
    return results
