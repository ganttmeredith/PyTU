"""
pytu/focused.py
===============
Focused transducer beam profile and associated thermal simulation.

Based on O'Neil's analytical model for spherically focused transducers.

Public API
----------
compute_focused_intensity(z, att_dB, I0, dz, transducer_radius, focal_depth,
                           frequency, sound_speed)
    Returns the focused beam intensity profile I_focused (W/cm²).

run_focused_thermal_simulation(z, I_focused, att_dB, kappa, density, c_p,
                                blood_perfusion, sonication_time, dt, dz_m)
    Returns T_focused and T_focused_history.
"""

import numpy as np
from tqdm import tqdm

from . import config as cfg
from .thermal import analyze_pancreas_heating


# ---------------------------------------------------------------------------
# Focused beam intensity (O'Neil model)
# ---------------------------------------------------------------------------

def compute_focused_intensity(z, att_dB, I0=None, dz=None,
                               transducer_radius=None, focal_depth=None,
                               frequency=None, sound_speed=None):
    """
    Compute on-axis intensity for a spherically focused transducer using
    O'Neil's analytical model, then apply cumulative tissue attenuation.

    The beam is normalised so that the peak at the focal point equals 4 × I0
    (a ~4× focused gain relative to the unfocused simulation).

    Parameters
    ----------
    z : ndarray              Depth grid (cm)
    att_dB : ndarray         Attenuation (dB/cm) at each depth
    I0 : float, optional     Surface intensity (W/cm²)
    dz : float, optional     Spatial step (cm)
    transducer_radius : float, optional  Transducer radius (cm)
    focal_depth : float, optional        Focal depth (cm)
    frequency : float, optional          Operating frequency (Hz)
    sound_speed : float, optional        Sound speed (m/s)

    Returns
    -------
    I_focused : ndarray   Focused beam intensity (W/cm²)
    """
    I0              = cfg.I0              if I0              is None else I0
    dz              = cfg.DZ              if dz              is None else dz
    transducer_radius = cfg.TRANSDUCER_RADIUS if transducer_radius is None else transducer_radius
    focal_depth     = cfg.FOCAL_DEPTH     if focal_depth     is None else focal_depth
    frequency       = cfg.FREQUENCY       if frequency       is None else frequency
    sound_speed     = cfg.SOUND_SPEED     if sound_speed     is None else sound_speed

    wavelength_cm = (sound_speed / frequency) * 100   # m → cm
    k = 2 * np.pi / wavelength_cm                     # wave number (cm⁻¹)

    I_focused = np.zeros_like(z)

    for i, zi in enumerate(z):
        if zi == 0:
            I_focused[i] = I0
        elif abs(zi / focal_depth - 1.0) < 1e-10:
            # At focal point — use the limiting expression
            I_focused[i] = I0 * (k * transducer_radius ** 2 / focal_depth) ** 2
        else:
            arg = k * transducer_radius ** 2 * (1 / zi - 1 / focal_depth) / 2
            I_focused[i] = I0 * (np.sin(arg)) ** 2

    # Apply cumulative tissue attenuation along the beam path
    for i in range(1, len(z)):
        delta_att = att_dB[i - 1] * dz
        I_focused[i] *= 10 ** (-delta_att / 10)

    # Normalise: focused peak ≈ 4× unfocused I0
    peak_idx = np.abs(z - focal_depth).argmin()
    if I_focused[peak_idx] > 0:
        normalization_factor = I0 * 4 / I_focused[peak_idx]
        I_focused *= normalization_factor

    return I_focused


# ---------------------------------------------------------------------------
# Focused thermal simulation
# ---------------------------------------------------------------------------

def run_focused_thermal_simulation(z, I_focused, att_dB, kappa, density, c_p,
                                    blood_perfusion, sonication_time=None,
                                    dt=None, dz_m=None, T_init=37.0):
    """
    Run the Pennes bioheat simulation driven by the focused beam heat source.

    Parameters mirror those in thermal.run_thermal_simulation; the only
    difference is that Q is computed from I_focused rather than the unfocused I.

    Returns
    -------
    T_focused : ndarray           Final temperature (°C)
    T_focused_history : ndarray   Shape (len(z), nt)
    """
    from .thermal import compute_heating_term, compute_stable_dt

    sonication_time = cfg.SONICATION_TIME if sonication_time is None else sonication_time
    dz_m = cfg.DZ / 100 if dz_m is None else dz_m
    dt = compute_stable_dt(kappa, dz_m) if dt is None else dt

    Q_focused, _ = compute_heating_term(att_dB, I_focused)
    nt = int(sonication_time / dt)

    T_focused = np.ones_like(z) * T_init
    T_focused_history = np.zeros((len(z), nt))
    T_focused_history[:, 0] = T_focused

    blood_den = cfg.BLOOD_DENSITY
    blood_cp  = cfg.BLOOD_SPECIFIC_HEAT
    blood_T   = cfg.BLOOD_TEMP

    print("Running thermal simulation for focused transducer...")
    for n in tqdm(range(1, nt)):
        dT = np.zeros_like(T_focused)
        for i in range(1, len(z) - 1):
            diffusion = kappa[i] * (T_focused[i + 1] - 2 * T_focused[i] + T_focused[i - 1]) / dz_m ** 2
            perfusion = (blood_perfusion[i] * blood_den * blood_cp
                         * (blood_T - T_focused[i]) / density[i] / c_p[i])
            source    = Q_focused[i] / (density[i] * c_p[i])
            dT[i]     = dt * (diffusion + perfusion + source)

        dT[0]  = dT[1]
        dT[-1] = dT[-2]
        T_focused += dT
        T_focused_history[:, n] = T_focused

    return T_focused, T_focused_history


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def run_focused_pipeline(z, att_dB, I_unfocused, kappa, density, c_p,
                          blood_perfusion, T_unfocused,
                          sonication_time=None, dt=None, dz_m=None):
    """
    Full focused transducer pipeline: compute intensity, run thermal sim,
    analyse pancreas heating, and return all results.

    Returns
    -------
    dict with keys:
        I_focused, T_focused, T_focused_history, pancreas_metrics
    """
    import pandas as pd
    from .comparison import direct_hitu_comparison

    sonication_time = cfg.SONICATION_TIME if sonication_time is None else sonication_time
    dz_m = cfg.DZ / 100 if dz_m is None else dz_m

    I_focused = compute_focused_intensity(z, att_dB)
    T_focused, T_focused_history = run_focused_thermal_simulation(
        z, I_focused, att_dB, kappa, density, c_p, blood_perfusion,
        sonication_time=sonication_time, dt=dt, dz_m=dz_m
    )
    metrics = analyze_pancreas_heating(z, T_focused, label="Focused Beam")

    # HITU comparison for focused results
    direct_hitu_comparison(T_focused, z)

    # Save focused vs unfocused comparison CSV
    pd.DataFrame({
        "Depth (cm)":      z,
        "Unfocused I (W/cm²)": I_unfocused,
        "Focused I (W/cm²)":   I_focused,
        "Unfocused T (°C)":    T_unfocused,
        "Focused T (°C)":      T_focused,
        "Difference (°C)":     T_focused - T_unfocused,
    }).to_csv("focused_vs_unfocused_comparison.csv", index=False)

    return {
        "I_focused":        I_focused,
        "T_focused":        T_focused,
        "T_focused_history": T_focused_history,
        "pancreas_metrics": metrics,
    }
