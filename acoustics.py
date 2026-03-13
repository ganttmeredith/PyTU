"""
pytu/acoustics.py
=================
Builds the depth grid and computes acoustic intensity / attenuation profiles.

Public API
----------
build_depth_grid(z_min, z_max, dz)
    Returns the 1-D depth array z (cm).

build_tissue_property_arrays(z, tissue_layers, frequency, perfusion_values, pancreas_ext)
    Populates per-point arrays for attenuation, thermal conductivity,
    specific heat, density, and blood perfusion.

compute_intensity_profile(z, att_dB, I0, dz)
    Computes the cumulative-attenuation intensity profile I(z).

skin_impact_summary(z, I, I0)
    Returns a dict of intensity checkpoints across the skin layers.
"""

import numpy as np
import pandas as pd

from . import config as cfg


# ---------------------------------------------------------------------------
# Grid construction
# ---------------------------------------------------------------------------

def build_depth_grid(z_min=None, z_max=None, dz=None):
    """
    Build the 1-D spatial depth grid.

    Parameters
    ----------
    z_min, z_max : float, optional
        Depth range in cm. Defaults to config.Z_MIN / Z_MAX.
    dz : float, optional
        Spatial resolution in cm. Defaults to config.DZ.

    Returns
    -------
    z : ndarray
        Depth array in cm.
    """
    z_min = cfg.Z_MIN if z_min is None else z_min
    z_max = cfg.Z_MAX if z_max is None else z_max
    dz = cfg.DZ if dz is None else dz
    return np.arange(z_min, z_max + dz, dz)


# ---------------------------------------------------------------------------
# Tissue property arrays
# ---------------------------------------------------------------------------

def build_tissue_property_arrays(z, tissue_layers=None, frequency=None,
                                  perfusion_values=None, pancreas_ext=None):
    """
    Populate per-depth-point tissue property arrays from the layer definitions.

    Parameters
    ----------
    z : ndarray
        Depth grid (cm).
    tissue_layers : list of dict, optional
        Layer definitions. Defaults to config.TISSUE_LAYERS.
    frequency : float, optional
        Operating frequency in Hz. Defaults to config.FREQUENCY.
    perfusion_values : dict, optional
        Blood perfusion values keyed by layer name. Defaults to config.PERFUSION_VALUES.
    pancreas_ext : dict, optional
        Properties for z >= last layer end. Defaults to config.PANCREAS_EXTENSION.

    Returns
    -------
    att_dB : ndarray        Attenuation coefficient (dB/cm) at each depth point
    k_thermal : ndarray     Thermal conductivity W/(m·K)
    c_p : ndarray           Specific heat J/(kg·K)
    density : ndarray       Density kg/m³
    blood_perfusion : ndarray  Perfusion ml_blood/(ml_tissue·s)
    """
    tissue_layers = tissue_layers or cfg.TISSUE_LAYERS
    frequency = frequency or cfg.FREQUENCY
    perfusion_values = perfusion_values or cfg.PERFUSION_VALUES
    pancreas_ext = pancreas_ext or cfg.PANCREAS_EXTENSION

    freq_MHz = frequency / 1e6

    att_dB = np.zeros_like(z)
    k_thermal = np.zeros_like(z)
    c_p = np.zeros_like(z)
    density = np.zeros_like(z)
    blood_perfusion = np.zeros_like(z)

    for layer in tissue_layers:
        idx = np.where((z >= layer["start"]) & (z < layer["end"]))
        att_dB[idx] = layer["attenuation"] * freq_MHz
        k_thermal[idx] = layer["thermal_conductivity"]
        c_p[idx] = layer["specific_heat"]
        density[idx] = layer["density"]
        blood_perfusion[idx] = perfusion_values.get(layer["name"], 0.0)

    # Extend pancreas properties beyond the last layer boundary
    last_end = tissue_layers[-1]["end"]
    ext_idx = np.where(z >= last_end)
    att_dB[ext_idx] = pancreas_ext["attenuation"] * freq_MHz
    k_thermal[ext_idx] = pancreas_ext["thermal_conductivity"]
    c_p[ext_idx] = pancreas_ext["specific_heat"]
    density[ext_idx] = pancreas_ext["density"]

    return att_dB, k_thermal, c_p, density, blood_perfusion


# ---------------------------------------------------------------------------
# Intensity profile
# ---------------------------------------------------------------------------

def compute_intensity_profile(z, att_dB, I0=None, dz=None):
    """
    Compute the on-axis acoustic intensity profile using cumulative attenuation.

    I(z) = I0 * 10^(-cumulative_att / 10)   where cumulative_att is in dB.

    Parameters
    ----------
    z : ndarray
        Depth grid (cm).
    att_dB : ndarray
        Attenuation coefficient (dB/cm) at each depth point.
    I0 : float, optional
        Peak intensity at the transducer surface (W/cm²). Defaults to config.I0.
    dz : float, optional
        Spatial step (cm). Defaults to config.DZ.

    Returns
    -------
    I : ndarray
        Acoustic intensity (W/cm²) at each depth point.
    """
    I0 = cfg.I0 if I0 is None else I0
    dz = cfg.DZ if dz is None else dz

    I = np.zeros_like(z)
    I[0] = I0
    cumulative_att = 0.0
    for i in range(1, len(z)):
        cumulative_att += att_dB[i - 1] * dz
        I[i] = I0 * 10 ** (-cumulative_att / 10)

    return I


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def skin_impact_summary(z, I, I0=None):
    """
    Summarise intensity transmission across the skin layers.

    Parameters
    ----------
    z : ndarray
        Depth grid (cm).
    I : ndarray
        Intensity profile (W/cm²).
    I0 : float, optional
        Surface intensity. Defaults to config.I0.

    Returns
    -------
    summary : dict
        Dict with intensity at each skin sublayer and total skin attenuation %.
    """
    I0 = cfg.I0 if I0 is None else I0
    summary = {
        "initial_intensity": I0,
        "intensity_after_epidermis": float(I[np.abs(z - 0.1).argmin()]),
        "intensity_after_dermis":    float(I[np.abs(z - 0.4).argmin()]),
        "intensity_after_hypodermis": float(I[np.abs(z - 1.0).argmin()]),
        "intensity_at_pancreas":     float(I[np.abs(z - 5.0).argmin()]),
    }
    summary["attenuation_by_skin_pct"] = (
        1 - summary["intensity_after_hypodermis"] / I0
    ) * 100
    return summary


def save_skin_impact_csv(summary, path="skin_impact_analysis.csv"):
    """Save the skin impact summary dict to a CSV file."""
    pd.DataFrame({
        "Metric": list(summary.keys()),
        "Value":  list(summary.values()),
    }).to_csv(path, index=False)


def save_simulation_results_csv(z, T, T_history, I, att_dB,
                                 path="enhanced_simulation_results.csv"):
    """Save the full simulation depth profile to CSV."""
    pd.DataFrame({
        "Depth_cm":               z,
        "Final_Temperature_C":    T,
        "Initial_Temperature_C":  T_history[:, 0],
        "Acoustic_Intensity_W_per_cm2": I,
        "Attenuation_dB_per_cm":  att_dB,
    }).to_csv(path, index=False)
