"""
PyTU Version 2.0 — Modular Therapeutic Ultrasound Simulation Package
============================================================
Modules:
    config          – shared constants and tissue definitions
    acoustics       – intensity profile and attenuation arrays
    thermal         – Pennes bioheat finite-difference solver
    focused         – focused transducer beam + thermal simulation
    pulsed          – duty-cycle / pulsed waveform simulations
    comparison      – HITU Simulator and Zderic model validation
    plotting        – all matplotlib figure helpers
    md_simulation   – MDAnalysis protein (diabetic) simulation (Part 2)
============================================================
"

from . import config, acoustics, thermal, focused, pulsed, comparison, plotting

try:
    from . import md_simulation
except ImportError:
    pass  # MDAnalysis is optional

__version__ = "2.0.0"
__author__ = "Gantt Meredith"
