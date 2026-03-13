"""
pytu/md_simulation.py
=====================
Part 2: Diabetic protein MD simulation using MDAnalysis.

Replaces the Colab-specific implementation (google.colab drive mounting,
!pip install) with a locally-runnable version.  PDB file paths are passed
as arguments rather than hard-coded Google Drive paths.

Public API
----------
load_protein(pdb_path)
    Load a PDB file into an MDAnalysis Universe.

apply_diabetic_modification(universe)
    Apply small random displacements to LYS/ARG residues to simulate glycation.

apply_ultrasound_effect(universe, frequency, duty_cycle, duration)
    Apply a pseudo-oscillatory displacement and compute RMSD over time.

run_md_simulation(pdb_kir6_2, pdb_irs1, output_dir, frequency,
                  duty_cycle, duration)
    Run the full Part 2 pipeline and save plots.

Requirements
------------
    MDAnalysis  (pip install MDAnalysis)
"""

import os
import numpy as np

try:
    import MDAnalysis as mda
    from MDAnalysis.analysis.rms import rmsd
    _MDA_AVAILABLE = True
except ImportError:
    _MDA_AVAILABLE = False
    mda = None
    rmsd = None


def _require_mda():
    if not _MDA_AVAILABLE:
        raise ImportError(
            "MDAnalysis is required for Part 2 (protein MD simulation). "
            "Install it with:  pip install MDAnalysis"
        )


# ---------------------------------------------------------------------------
# Protein loading
# ---------------------------------------------------------------------------

def load_protein(pdb_path: str):
    """
    Load a PDB file into an MDAnalysis Universe.

    Parameters
    ----------
    pdb_path : str  Path to the PDB file.

    Returns
    -------
    universe : mda.Universe
    """
    _require_mda()
    if not os.path.isfile(pdb_path):
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")
    return mda.Universe(pdb_path)


# ---------------------------------------------------------------------------
# Diabetic modification (glycation simulation)
# ---------------------------------------------------------------------------

def apply_diabetic_modification(universe):
    """
    Simulate glycation by applying small random displacements to Lysine (LYS)
    and Arginine (ARG) residues.

    Parameters
    ----------
    universe : mda.Universe

    Returns
    -------
    diabetic_universe : mda.Universe   Modified copy
    """
    _require_mda()
    diabetic_universe = universe.copy()
    glycated = diabetic_universe.select_atoms("resname LYS ARG")
    for atom in glycated.atoms:
        atom.position += np.random.normal(0, 0.1, size=3)
    return diabetic_universe


# ---------------------------------------------------------------------------
# Ultrasound effect
# ---------------------------------------------------------------------------

def apply_ultrasound_effect(universe, frequency=1e6, duty_cycle=1.0, duration=300):
    """
    Apply a pseudo-oscillatory atomic displacement to simulate acoustic
    perturbation and measure RMSD vs. the original positions at each time step.

    Parameters
    ----------
    universe : mda.Universe
    frequency : float   Ultrasound frequency (Hz). Default 1 MHz.
    duty_cycle : float  Duty cycle [0, 1]. Default 1.0 (continuous).
    duration : int      Number of time steps (seconds). Default 300.

    Returns
    -------
    rmsd_changes : list of float
        RMSD (Å) between the perturbed and original positions at each step.
    """
    _require_mda()
    rmsd_changes = []
    original_positions = universe.atoms.positions.copy()

    for t in range(int(duration)):
        displacement = np.sin(2 * np.pi * frequency * t) * duty_cycle * 0.01
        universe.atoms.positions += displacement
        rmsd_value = rmsd(universe.atoms.positions, original_positions)
        rmsd_changes.append(rmsd_value)
        universe.atoms.positions = original_positions.copy()

    return rmsd_changes


# ---------------------------------------------------------------------------
# Full Part 2 pipeline
# ---------------------------------------------------------------------------

def run_md_simulation(pdb_kir6_2: str, pdb_irs1: str,
                       output_dir: str = ".",
                       frequency: float = 1e6,
                       duty_cycle: float = 1.0,
                       duration: int = 300):
    """
    Full Part 2 pipeline: load proteins, apply diabetic modifications,
    run ultrasound simulations, and save RMSD plots.

    Parameters
    ----------
    pdb_kir6_2 : str   Path to Kir6.2 PDB file
    pdb_irs1 : str     Path to IRS1 PDB file
    output_dir : str   Directory for output plots
    frequency : float  Ultrasound frequency (Hz). Default 1 MHz.
    duty_cycle : float Duty cycle [0, 1]. Default 1.0 (continuous).
    duration : int     Simulation duration (s). Default 300.
    """
    from .plotting import plot_rmsd_changes

    _require_mda()
    os.makedirs(output_dir, exist_ok=True)

    print("\n=== Part 2: Diabetic MD Simulation ===")
    print(f"  Kir6.2 PDB:  {pdb_kir6_2}")
    print(f"  IRS1 PDB:    {pdb_irs1}")
    print(f"  Frequency:   {frequency / 1e6:.1f} MHz")
    print(f"  Duty cycle:  {duty_cycle * 100:.0f}%")
    print(f"  Duration:    {duration} s")

    # Load proteins
    kir6_2 = load_protein(pdb_kir6_2)
    irs1   = load_protein(pdb_irs1)

    # Create diabetic versions
    diabetic_kir6_2 = apply_diabetic_modification(kir6_2)
    diabetic_irs1   = apply_diabetic_modification(irs1)

    # Run ultrasound simulations
    print("\nRunning simulation on Kir6_2 and IRS1 proteins...")
    rmsd_kir_d  = apply_ultrasound_effect(diabetic_kir6_2, frequency, duty_cycle, duration)
    rmsd_kir_nd = apply_ultrasound_effect(kir6_2,          frequency, duty_cycle, duration)
    rmsd_irs1_d  = apply_ultrasound_effect(diabetic_irs1,  frequency, duty_cycle, duration)
    rmsd_irs1_nd = apply_ultrasound_effect(irs1,           frequency, duty_cycle, duration)

    # Plot
    time = np.arange(duration)
    plot_rmsd_changes(time, rmsd_kir_d, rmsd_kir_nd, rmsd_irs1_d, rmsd_irs1_nd,
                      output_dir=output_dir)

    print("\nPart 2 complete.")
    return {
        "rmsd_kir_diabetic":     rmsd_kir_d,
        "rmsd_kir_nondiabetic":  rmsd_kir_nd,
        "rmsd_irs1_diabetic":    rmsd_irs1_d,
        "rmsd_irs1_nondiabetic": rmsd_irs1_nd,
    }
