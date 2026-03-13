"""
pytu/config.py
==============
Shared simulation constants and tissue definitions for PyTU v2.

All other modules import from here so parameters stay in one place.
To adjust simulation parameters (e.g., frequency, intensity, depth,
tissue layout), edit this file only.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Acoustic parameters
# ---------------------------------------------------------------------------
FREQUENCY = 1e6          # Hz  (1 MHz)
SOUND_SPEED = 1500       # m/s (default soft tissue)
WAVELENGTH = SOUND_SPEED / FREQUENCY   # m

# Spatial grid
Z_MIN = 0                # cm
Z_MAX = 10               # cm
DZ = 0.01                # cm  (spatial resolution)

# Peak acoustic intensity at transducer surface
I0 = 5                   # W/cm²

# ---------------------------------------------------------------------------
# Tissue layer definitions
# ---------------------------------------------------------------------------
# Each entry describes one tissue layer from skin surface inward.
# Values derived from Saab et al. (J Ultrasound Med 2023) and literature.
TISSUE_LAYERS = [
    {
        "name": "Epidermis",
        "start": 0,
        "end": 0.1,
        "attenuation": 4.2,        # dB/(cm·MHz)
        "density": 1120,           # kg/m³
        "sound_speed": 1624,       # m/s
        "thermal_conductivity": 0.235,   # W/(m·K)
        "specific_heat": 3600,           # J/(kg·K)
    },
    {
        "name": "Dermis",
        "start": 0.1,
        "end": 0.4,
        "attenuation": 3.5,
        "density": 1109,
        "sound_speed": 1624,
        "thermal_conductivity": 0.445,
        "specific_heat": 3300,
    },
    {
        "name": "Hypodermis",
        "start": 0.4,
        "end": 1.0,
        "attenuation": 2.1,
        "density": 971,
        "sound_speed": 1450,
        "thermal_conductivity": 0.185,
        "specific_heat": 2700,
    },
    {
        "name": "Muscle",
        "start": 1,
        "end": 2,
        "attenuation": 1.09,
        "density": 1050,
        "sound_speed": 1547,
        "thermal_conductivity": 0.51,
        "specific_heat": 3421,
    },
    {
        "name": "Visceral Fat",
        "start": 2,
        "end": 3,
        "attenuation": 0.63,
        "density": 950,
        "sound_speed": 1478,
        "thermal_conductivity": 0.21,
        "specific_heat": 2348,
    },
    {
        "name": "Mixed Tissue",
        "start": 3,
        "end": 4,
        "attenuation": 0.85,
        "density": 1000,
        "sound_speed": 1500,
        "thermal_conductivity": 0.40,
        "specific_heat": 3500,
    },
    {
        "name": "Pancreas",
        "start": 4,
        "end": 6.5,
        "attenuation": 0.85,
        "density": 1045,
        "sound_speed": 1514,
        "thermal_conductivity": 0.51,
        "specific_heat": 3164,
    },
]

# Properties used for z >= 6.5 cm (continuation of pancreas)
PANCREAS_EXTENSION = {
    "attenuation": 0.85,
    "thermal_conductivity": 0.51,
    "specific_heat": 3164,
    "density": 1045,
}

# ---------------------------------------------------------------------------
# Blood perfusion parameters (Pennes bioheat model)
# ---------------------------------------------------------------------------
# ml_Blood / ml_tissue / s
PERFUSION_VALUES = {
    "Epidermis":    0.0,    # no vessels in epidermis
    "Dermis":       0.013,
    "Hypodermis":   0.001,
    "Muscle":       0.027,
    "Visceral Fat": 0.001,
    "Mixed Tissue": 0.018,
    "Pancreas":     0.038,
}

BLOOD_TEMP = 37.0         # °C
BLOOD_DENSITY = 1060      # kg/m³
BLOOD_SPECIFIC_HEAT = 3770  # J/(kg·K)

# ---------------------------------------------------------------------------
# Thermal simulation parameters
# ---------------------------------------------------------------------------
SONICATION_TIME = 300     # seconds
DT_DEFAULT = 0.01         # time step (s) — overridden by stability criterion

# ---------------------------------------------------------------------------
# Focused transducer parameters
# ---------------------------------------------------------------------------
TRANSDUCER_RADIUS = 1.5   # cm
FOCAL_DEPTH = 3.96        # cm

# ---------------------------------------------------------------------------
# Pancreas depth range
# ---------------------------------------------------------------------------
PANCREAS_START = 4.0      # cm
PANCREAS_END = 6.5        # cm

# ---------------------------------------------------------------------------
# Zderic reference data (estimated from Zderic's paper, Figure 4)
# ---------------------------------------------------------------------------
ZDERIC_DATA = {
    "duration_min": [1, 2, 3, 4, 5],
    "pancreas_temp_10pct":  [37.1, 37.2, 37.3, 37.4, 37.5],
    "pancreas_temp_25pct":  [37.3, 37.5, 37.7, 37.8, 38.0],
    "pancreas_temp_50pct":  [37.5, 38.0, 38.3, 38.5, 38.8],
    "pancreas_temp_100pct": [38.3, 39.0, 39.5, 40.1, 40.5],
    "overall_temp_10pct":   [40,   43,   45,   47,   49],
    "overall_temp_25pct":   [43,   49,   52,   55.0, 59.0],
    "overall_temp_50pct":   [50,   60,   68,   75.0, 82.0],
    "overall_temp_100pct":  [65.0, 80.0, 100.0, 115.0, 125.0],
    "obese_pancreas_temp_10pct":  [37.2, 37.4, 37.7, 38.0, 38.2],
    "obese_pancreas_temp_25pct":  [38.0, 38.4, 38.9, 39.2, 39.7],
    "obese_pancreas_temp_50pct":  [38.7, 39.8, 40.8, 41.5, 42.2],
    "obese_pancreas_temp_100pct": [40.5, 42.8, 44.8, 46.0, 47.2],
    "obese_overall_temp_10pct":   [42,   46,   49,   51,   52],
    "obese_overall_temp_25pct":   [51,   56,   65,   75.0, 80.0],
    "obese_overall_temp_50pct":   [70,   75,   98,   110,  125.0],
    "obese_overall_temp_100pct":  [105, 125.0, 155.0, 180.0, 215.0],
}

# HITU Simulator reference data (Zderic paper figures)
HITU_Z = np.array([0, 1, 2, 3, 4, 5, 6, 7])
HITU_TEMP = np.array([37.0, 37.1, 37.25, 37.55, 37.35, 37.3, 37.25, 37.2])
