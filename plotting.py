"""
pytu/plotting.py
================
All matplotlib figure helpers for PyTU v2.

Each function creates, saves (to disk), and closes a figure.
Call matplotlib.use("Agg") before importing if running headlessly.

Public API
----------
plot_intensity_vs_depth(z, I, tissue_layers, I0, save_path)
plot_temperature_distribution(z, T, T_history, tissue_layers,
                               sonication_time, save_path)
plot_temperature_evolution(time_points, T_history, target_idx, save_path)
plot_focused_beam_intensity(z, I_focused, I, tissue_layers, focal_depth,
                             I0, save_path)
plot_focused_temperature_distribution(z, T_focused, T, tissue_layers,
                                       focal_depth, sonication_time, save_path)
plot_pulsed_duty_curve(z, T_duty, T, duty_factor, duration_min,
                        tissue_layers, save_path)
plot_rmsd_changes(time, rmsd_kir, rmsd_kir_nd, rmsd_irs1, rmsd_irs1_nd,
                   output_dir)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import config as cfg


def _add_tissue_labels(ax, tissue_layers, y_top, fontsize=9):
    """Helper: add vertical layer boundaries and centred text labels."""
    for layer in tissue_layers:
        ax.axvline(x=layer["start"], color="gray", linestyle=":", alpha=0.7)
        mid = layer["start"] + (layer["end"] - layer["start"]) / 2
        ax.text(mid, y_top, layer["name"], rotation=90, va="top",
                ha="center", fontsize=fontsize)


# ---------------------------------------------------------------------------
# Part 1 — TUS plots
# ---------------------------------------------------------------------------

def plot_intensity_vs_depth(z, I, tissue_layers=None, I0=None,
                             save_path="intensity_vs_depth_enhanced.png"):
    """Acoustic intensity vs. depth with tissue layer boundaries."""
    tissue_layers = tissue_layers or cfg.TISSUE_LAYERS
    I0 = cfg.I0 if I0 is None else I0

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(z, I, "b-", linewidth=2, label="Acoustic Intensity (W/cm²)")
    _add_tissue_labels(ax, tissue_layers, I0 * 0.9)
    ax.set_xlabel("Depth (cm)", fontsize=12)
    ax.set_ylabel("Intensity (W/cm²)", fontsize=12)
    ax.set_title("Acoustic Intensity vs Depth with Enhanced Skin Model", fontsize=14)
    ax.grid(True)
    ax.legend(fontsize=12)
    fig.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_temperature_distribution(z, T, T_history, tissue_layers=None,
                                   sonication_time=None,
                                   save_path="temperature_distribution_enhanced.png"):
    """Final and initial temperature distribution with tissue layer boundaries."""
    tissue_layers = tissue_layers or cfg.TISSUE_LAYERS
    sonication_time = cfg.SONICATION_TIME if sonication_time is None else sonication_time

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(z, T, "r-", linewidth=2, label="Final Temperature")
    ax.plot(z, T_history[:, 0], "b--", linewidth=1.5, label="Initial Temperature")
    _add_tissue_labels(ax, tissue_layers, ax.get_ylim()[1] - 0.2)
    ax.set_xlabel("Depth (cm)", fontsize=12)
    ax.set_ylabel("Temperature (°C)", fontsize=12)
    ax.set_title(
        f"Temperature Distribution After {sonication_time}s Sonication\n"
        "With Enhanced Skin Model & Perfusion",
        fontsize=14,
    )
    ax.grid(True)
    ax.legend(fontsize=12)
    fig.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_temperature_evolution(time_points, T_history, target_idx,
                                sonication_time=None,
                                save_path="temperature_evolution_5cm_enhanced.png"):
    """Temperature vs. time at a single target depth index."""
    sonication_time = cfg.SONICATION_TIME if sonication_time is None else sonication_time

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time_points, T_history[target_idx, :], "r-", linewidth=2)
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Temperature (°C)", fontsize=12)
    ax.set_title(
        "Temperature Evolution at Target Depth (5 cm)\nWith Enhanced Skin Model",
        fontsize=14,
    )
    ax.grid(True)
    fig.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_focused_beam_intensity(z, I_focused, I, tissue_layers=None,
                                 focal_depth=None, I0=None,
                                 save_path="focused_beam_intensity.png"):
    """Focused vs. unfocused intensity profile."""
    tissue_layers = tissue_layers or cfg.TISSUE_LAYERS
    focal_depth = cfg.FOCAL_DEPTH if focal_depth is None else focal_depth
    I0 = cfg.I0 if I0 is None else I0

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(z, I_focused, "r-", linewidth=2, label="Focused Beam Intensity")
    ax.plot(z, I, "b--", linewidth=1.5, label="Unfocused Beam (for comparison)")
    ax.axvline(x=focal_depth, color="green", linestyle=":",
               label=f"Focal Point: {focal_depth} cm")
    _add_tissue_labels(ax, tissue_layers, I0 * 5.5)
    ax.set_xlabel("Depth (cm)", fontsize=12)
    ax.set_ylabel("Intensity (W/cm²)", fontsize=12)
    ax.set_title("Focused vs Unfocused Beam Intensity Profile", fontsize=14)
    ax.grid(True)
    ax.legend(fontsize=12)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_focused_temperature_distribution(z, T_focused, T, tissue_layers=None,
                                           focal_depth=None, sonication_time=None,
                                           save_path="focused_temperature_distribution.png"):
    """Focused vs unfocused final temperature distributions."""
    tissue_layers = tissue_layers or cfg.TISSUE_LAYERS
    focal_depth = cfg.FOCAL_DEPTH if focal_depth is None else focal_depth
    sonication_time = cfg.SONICATION_TIME if sonication_time is None else sonication_time

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(z, T_focused, "r-", linewidth=2, label="Focused Beam Temperature")
    ax.plot(z, T, "b--", linewidth=1.5, label="Unfocused Beam Temperature")
    ax.axvline(x=focal_depth, color="green", linestyle=":",
               label=f"Focal Point: {focal_depth} cm")
    ax.set_xlabel("Depth (cm)", fontsize=12)
    ax.set_ylabel("Temperature (°C)", fontsize=12)
    ax.set_title(
        f"Temperature Distribution After {sonication_time}s Sonication: "
        "Focused vs Unfocused",
        fontsize=14,
    )
    ax.grid(True)
    ax.legend(fontsize=12)
    for layer in tissue_layers:
        ax.axvline(x=layer["start"], color="gray", linestyle=":", alpha=0.7)
    fig.savefig(save_path, dpi=900, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_pulsed_duty_curve(z, T_duty, T, duty_factor, duration_min,
                            tissue_layers=None,
                            save_path=None):
    """Temperature distribution for a single duty-factor run."""
    tissue_layers = tissue_layers or cfg.TISSUE_LAYERS
    if save_path is None:
        save_path = f"temp_distribution_{int(duty_factor * 100)}pct_duty.png"

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(z, T_duty, "r-", linewidth=2, label=f"{duty_factor * 100:.0f}% Duty")
    ax.plot(z, T, "b--", linewidth=1.5, label="Original Simulation")
    for layer in tissue_layers:
        ax.axvline(x=layer["start"], color="gray", linestyle=":", alpha=0.5)
    ax.axvspan(cfg.PANCREAS_START, cfg.PANCREAS_END, color="yellow", alpha=0.2,
               label="Pancreas Region")
    ax.set_xlabel("Depth (cm)", fontsize=12)
    ax.set_ylabel("Temperature (°C)", fontsize=12)
    ax.set_title(
        f"Temperature Distribution: {duty_factor * 100:.0f}% Duty Factor, "
        f"{duration_min} min",
        fontsize=14,
    )
    ax.grid(True)
    ax.legend(fontsize=12)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


# ---------------------------------------------------------------------------
# Part 2 — MD simulation plots
# ---------------------------------------------------------------------------

def plot_rmsd_changes(time, rmsd_kir_d, rmsd_kir_nd, rmsd_irs1_d, rmsd_irs1_nd,
                       output_dir="."):
    """RMSD vs. time for Kir6.2 and IRS1 diabetic / non-diabetic conditions."""
    os.makedirs(output_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ax1.plot(time, rmsd_kir_d,  label="Diabetic Kir6.2",     color="red")
    ax1.plot(time, rmsd_kir_nd, label="Non-Diabetic Kir6.2", color="blue")
    ax1.set_title("Kir6.2 RMSD Changes under Ultrasound")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("RMSD (nm)")
    ax1.legend()

    ax2.plot(time, rmsd_irs1_d,  label="Diabetic IRS1",     color="red")
    ax2.plot(time, rmsd_irs1_nd, label="Non-Diabetic IRS1", color="blue")
    ax2.set_title("IRS1 RMSD Changes under Ultrasound")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("RMSD (nm)")
    ax2.legend()

    fig.tight_layout()
    kir_path  = os.path.join(output_dir, "Kir6_2_RMSD_Changes.png")
    irs1_path = os.path.join(output_dir, "IRS1_RMSD_Changes.png")
    fig.savefig(kir_path, dpi=300, bbox_inches="tight")
    # Save full figure (both subplots) under both names for compatibility
    fig.savefig(irs1_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved RMSD plots to {output_dir}")
