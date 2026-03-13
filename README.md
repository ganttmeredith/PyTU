# PyTU v2 — Therapeutic Ultrasound Simulation

> **GM_2026_03_06:** PyTU Version 1 was a single-script Colab notebook uploaded to GitHub prior to initial publication of the applied clinical work. **PyTU Version 2** is a fully modular Python package, designed for clinicians and sonographers who need to adjust tissue type, target depth, transducer geometry, and sonication parameters to their own use case. By modelling acoustic attenuation and thermal dose at both the organ and protein level, PyTU serves as a stepping stone in image-guided therapeutics.

Diabetes mellitus is a disease of the pancreas, marked by insufficient processing of insulin from the beta cells of the islets of Langherans. Key mechanosensitive proteins found in these cells, as evidenced in vitro and in vivo under the work of Dr. Vesna Zderic, are modulated by therapeutic ultrasound. While the direct clinical application of therapeutic ultrasound for insulin release in the pancreas is explored, a key first-step is to ensure that the acoustic and thermal dose delivered to the target tissue is accurate. This package serves as a tool to model the acoustic and thermal dose delivered to the target tissue given a unique sonication protocol. 

---

## Package Structure

```
PyTU/
├── main.py                 
├── requirements.txt
└── pytu/
    ├── config.py            ← all constants & tissue definitions
    ├── acoustics.py         ← intensity profile, attenuation arrays
    ├── thermal.py           ← Pennes bioheat FD solver
    ├── focused.py           ← focused transducer (O'Neil model)
    ├── pulsed.py            ← duty-cycle / pulsed simulations
    ├── comparison.py        ← HITU Simulator & Zderic validation
    ├── plotting.py          ← all matplotlib helpers
    └── md_simulation.py     ← MDAnalysis protein simulation (Part 2)
```

---

## Installation

```bash
pip install -r requirements.txt
```

> **Part 2 (protein MD simulation)** additionally requires `MDAnalysis` and local PDB files for Kir6.2 and IRS1. If these are not available, use `--skip-md`.

---

## Usage

## Part 1 only (TUS acoustic + thermal modelling)

```bash
python main.py --skip-md --output-dir ./outputs/
```

## Part 1 + Part 2 (requires MDAnalysis and PDB files)

```bash
python main.py \
    --pdb-kir6-2 /path/to/Kir6_2.pdb \
    --pdb-irs1   /path/to/IRS1.pdb   \
    --output-dir ./outputs/
```

## All CLI flags

| Flag | Default | Description |

| `--skip-md` | — | Skip Part 2 |
| `--output-dir DIR` | `./outputs` | Output CSV/PNG directory |
| `--pdb-kir6-2 PATH` | — | Kir6.2 PDB file |
| `--pdb-irs1 PATH` | — | IRS1 PDB file |
| `--sonication-time S` | 300 | Override sonication duration (s) |
| `--intensity W` | 5 | Override peak intensity (W/cm²) |

---

## Simulation Parameters

All shared constants reside in `pytu/config.py` : tissue layers, blood perfusion values, Zderic/HITU reference data, focused transducer geometry, and grid resolution. This file is the main work horse for amending the simulation protocol for your clinical application.

---

## Outputs

All output files are written to `--output-dir`.

| File | Description |
|---|---|
| `intensity_vs_depth_enhanced.png` | Acoustic intensity vs depth |
| `temperature_distribution_enhanced.png` | Final temperature profile |
| `temperature_evolution_5cm_enhanced.png` | Temp vs time at 5 cm depth |
| `focused_beam_intensity.png` | Focused vs unfocused intensity |
| `focused_temperature_distribution.png` | Focused vs unfocused final temp |
| `hitu_direct_comparison.png/.csv` | HITU Simulator validation |
| `zderic_model_comparison.png` | Zderic single-point comparison |
| `zderic_full_comparison.png/.csv` | Zderic multi-timepoint comparison |
| `duty_factor_comparison_onshape_python.png` | Multi-duty sweep |
| `skin_impact_analysis.csv` | Skin attenuation summary |
| `enhanced_simulation_results.csv` | Full depth-profile data |
| `pancreas_thermal_analysis.csv` | Pancreas heating statistics |
| `Kir6_2_RMSD_Changes.png` | Kir6.2 RMSD plot (Part 2) |
| `IRS1_RMSD_Changes.png` | IRS1 RMSD plot (Part 2) |


## For a more in-depth overview of the clinical motivations, applications, and complete use case of PyTU, please refer to the dissertation of Dr. Gantt Meredith, https://www.proquest.com/docview/3281893223?pq-origsite=gscholar&fromopenview=true&sourcetype=Dissertations%20&%20Theses


