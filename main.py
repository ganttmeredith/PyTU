"""
main.py  —  PyTU v2 Entry Point
================================
Orchestrates the full PyTU pipeline.

Usage
-----
# Run Part 1 only (TUS acoustic + thermal modeling):
    python main.py --skip-md --output-dir ./outputs/

# Run Part 1 + Part 2 (requires MDAnalysis and PDB files):
    python main.py \\
        --pdb-kir6-2 /path/to/Kir6_2.pdb \\
        --pdb-irs1   /path/to/IRS1.pdb   \\
        --output-dir ./outputs/

Optional flags:
    --skip-md         Skip Part 2 (MD protein simulation)
    --output-dir DIR  Directory for all output CSV and PNG files (default: ./outputs)
    --pdb-kir6-2 PATH Path to Kir6.2 PDB file
    --pdb-irs1   PATH Path to IRS1 PDB file
    --sonication-time SECONDS  Override default sonication duration (default: 300 s)
    --intensity I0    Override peak acoustic intensity in W/cm² (default: 5)
"""

import argparse
import os
import sys
import numpy as np

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="PyTU v2 — Therapeutic Ultrasound Simulation"
    )
    parser.add_argument(
        "--skip-md", action="store_true",
        help="Skip Part 2 (requires MDAnalysis and PDB files)."
    )
    parser.add_argument(
        "--output-dir", default="./outputs",
        help="Directory for all CSV and PNG outputs. Default: ./outputs"
    )
    parser.add_argument("--pdb-kir6-2", default=None, metavar="PATH",
                        help="Path to Kir6.2 PDB file (Part 2).")
    parser.add_argument("--pdb-irs1",   default=None, metavar="PATH",
                        help="Path to IRS1 PDB file (Part 2).")
    parser.add_argument("--sonication-time", type=float, default=None,
                        help="Sonication duration in seconds (default: 300).")
    parser.add_argument("--intensity", type=float, default=None, dest="I0",
                        help="Peak acoustic intensity in W/cm² (default: 5).")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Part 1: TUS Acoustic & Thermal Modeling
# ---------------------------------------------------------------------------

def run_part1(output_dir, sonication_time=None, I0=None):
    """
    Run the full Part 1 TUS pipeline:
      1. Build depth grid and tissue property arrays
      2. Compute unfocused acoustic intensity profile
      3. Run unfocused Pennes bioheat simulation
      4. Compare with HITU Simulator and Zderic model
      5. Focused transducer simulation
      6. Multi-duty-factor pulsed simulations
    """
    from pytu import config as cfg
    from pytu import acoustics, thermal, focused, comparison
    from pytu import plotting

    # Override config values if supplied via CLI
    if I0 is not None:
        cfg.I0 = I0
    if sonication_time is not None:
        cfg.SONICATION_TIME = sonication_time

    os.makedirs(output_dir, exist_ok=True)

    def out(filename):
        """Return a full output path."""
        return os.path.join(output_dir, filename)

    print("\n" + "=" * 60)
    print("  PyTU v2  —  Part 1: TUS Acoustic & Thermal Modeling")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Build grid and tissue property arrays
    # ------------------------------------------------------------------
    print("\n[1/7] Building depth grid and tissue property arrays...")
    z = acoustics.build_depth_grid()
    att_dB, k_thermal, c_p, density, blood_perfusion = (
        acoustics.build_tissue_property_arrays(z)
    )
    dz_m = cfg.DZ / 100   # cm → m

    # ------------------------------------------------------------------
    # 2. Acoustic intensity profile (unfocused)
    # ------------------------------------------------------------------
    print("[2/7] Computing unfocused acoustic intensity profile...")
    I = acoustics.compute_intensity_profile(z, att_dB)

    plotting.plot_intensity_vs_depth(
        z, I, save_path=out("intensity_vs_depth_enhanced.png")
    )

    # Skin impact analysis
    skin_summary = acoustics.skin_impact_summary(z, I)
    acoustics.save_skin_impact_csv(skin_summary, path=out("skin_impact_analysis.csv"))
    print("\nSkin Impact Analysis:")
    print(f"  Initial intensity:              {skin_summary['initial_intensity']:.2f} W/cm²")
    print(f"  After skin (hypodermis):        {skin_summary['intensity_after_hypodermis']:.2f} W/cm²")
    print(f"  Skin attenuation:               {skin_summary['attenuation_by_skin_pct']:.1f}%")

    # ------------------------------------------------------------------
    # 3. Unfocused Pennes bioheat simulation
    # ------------------------------------------------------------------
    print("\n[3/7] Running unfocused thermal simulation...")
    Q, alpha_Np = thermal.compute_heating_term(att_dB, I)
    kappa = thermal.compute_thermal_diffusivity(k_thermal, density, c_p)
    dt = thermal.compute_stable_dt(kappa, dz_m)

    T, T_history, time_points = thermal.run_thermal_simulation(
        z, Q, kappa, density, c_p, blood_perfusion,
        sonication_time=cfg.SONICATION_TIME, dt=dt, dz_m=dz_m
    )

    target_depth_idx = np.abs(z - 5.0).argmin()
    print(f"  Max temperature at target depth (5 cm): "
          f"{T_history[target_depth_idx, -1]:.1f}°C")

    plotting.plot_temperature_distribution(
        z, T, T_history, sonication_time=cfg.SONICATION_TIME,
        save_path=out("temperature_distribution_enhanced.png")
    )
    plotting.plot_temperature_evolution(
        time_points, T_history, target_depth_idx,
        sonication_time=cfg.SONICATION_TIME,
        save_path=out("temperature_evolution_5cm_enhanced.png")
    )

    # Save full simulation CSV
    acoustics.save_simulation_results_csv(
        z, T, T_history, I, att_dB,
        path=out("enhanced_simulation_results.csv")
    )

    # ------------------------------------------------------------------
    # 4. Pancreas heating analysis + HITU / Zderic comparisons
    # ------------------------------------------------------------------
    print("\n[4/7] Analysing pancreas heating and running comparisons...")
    pancreas_metrics = thermal.analyze_pancreas_heating(z, T)
    thermal.save_pancreas_analysis_csv(
        pancreas_metrics, path=out("pancreas_thermal_analysis.csv")
    )

    comparison.direct_hitu_comparison(
        T, z,
        save_csv=out("hitu_direct_comparison.csv"),
        save_png=out("hitu_direct_comparison.png")
    )

    comparison.compare_with_zderic_model(
        z, T, T_history, time_points, cfg.SONICATION_TIME,
        save_png=out("zderic_model_comparison.png")
    )

    comparison.compare_with_zderic_model_full(
        z, T_history, time_points, cfg.SONICATION_TIME,
        save_png=out("zderic_full_comparison.png"),
        save_csv=out("zderic_full_comparison.csv")
    )

    # ------------------------------------------------------------------
    # 5. Focused transducer simulation
    # ------------------------------------------------------------------
    print("\n[5/7] Running focused transducer simulation...")
    from pytu.focused import run_focused_pipeline
    focused_results = run_focused_pipeline(
        z, att_dB, I, kappa, density, c_p, blood_perfusion, T,
        sonication_time=cfg.SONICATION_TIME, dt=dt, dz_m=dz_m
    )
    I_focused  = focused_results["I_focused"]
    T_focused  = focused_results["T_focused"]

    plotting.plot_focused_beam_intensity(
        z, I_focused, I, save_path=out("focused_beam_intensity.png")
    )
    plotting.plot_focused_temperature_distribution(
        z, T_focused, T, sonication_time=cfg.SONICATION_TIME,
        save_path=out("focused_temperature_distribution.png")
    )
    thermal.save_pancreas_analysis_csv(
        focused_results["pancreas_metrics"],
        path=out("focused_pancreas_thermal_analysis.csv")
    )

    # ------------------------------------------------------------------
    # 6. Zderic full2 multi-duty comparison
    # ------------------------------------------------------------------
    print("\n[6/7] Running Zderic multi-duty-factor comparison...")
    comparison.compare_with_zderic_model_full2(
        z, Q, kappa, density, c_p, blood_perfusion, dz_m, dt,
        save_png=out("duty_factor_comparison_onshape_python.png")
    )

    print("\n[7/7] Part 1 complete. All outputs saved to:", output_dir)
    return z, Q, kappa, density, c_p, blood_perfusion, dz_m, dt


# ---------------------------------------------------------------------------
# Part 2: Diabetic MD Simulation
# ---------------------------------------------------------------------------

def run_part2(pdb_kir6_2, pdb_irs1, output_dir):
    """Run the MDAnalysis diabetic protein simulation pipeline."""
    try:
        from pytu.md_simulation import run_md_simulation
    except ImportError as e:
        print(f"\n[Part 2] Skipped — {e}")
        return

    print("\n" + "=" * 60)
    print("  PyTU v2  —  Part 2: Diabetic MD Protein Simulation")
    print("=" * 60)

    run_md_simulation(
        pdb_kir6_2=pdb_kir6_2,
        pdb_irs1=pdb_irs1,
        output_dir=output_dir,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Run Part 1
    run_part1(
        output_dir=args.output_dir,
        sonication_time=args.sonication_time,
        I0=args.I0,
    )

    # Run Part 2 unless skipped
    if not args.skip_md:
        if args.pdb_kir6_2 is None or args.pdb_irs1 is None:
            print(
                "\n[Part 2] Skipped — use --pdb-kir6-2 and --pdb-irs1 to "
                "provide PDB files, or pass --skip-md to suppress this message."
            )
        else:
            run_part2(
                pdb_kir6_2=args.pdb_kir6_2,
                pdb_irs1=args.pdb_irs1,
                output_dir=args.output_dir,
            )

    print("\nPyTU v2 simulation complete.\n")


if __name__ == "__main__":
    main()
