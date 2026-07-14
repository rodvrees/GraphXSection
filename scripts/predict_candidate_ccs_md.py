#!/usr/bin/env python3
"""
Independent, physics-based candidate-to-peak matching check for Step 2 of
protomer_ccs_predictor_roadmap.md.

build_protomer_labels.py's `match_candidates_to_peaks()` pairs each molecule's
candidates (ranked by proton-affinity proxy) to its observed CCS peaks (ranked
by intensity) *positionally* -- rank-1 candidate to rank-1 peak, and so on.
That is an explicitly documented, unvalidated heuristic (see that script's own
docstring): it assumes the most stable protonation site produces the most
intense peak, which is plausible but unconfirmed. GraphXSection_torch can't
supply an independent check either -- it was never trained on protomer pairs
and is known to underestimate the CCS spread between them.

This script computes a genuinely independent, ORCA/MOBCAL-based predicted CCS
for a candidate, so `validate_matching.py` can re-derive the candidate<->peak
assignment by nearest-CCS matching (the roadmap's originally-intended
approach) and compare it against the positional heuristic, on a small
validation subset -- NOT the full dataset (see the per-candidate cost note
below).

Pipeline per candidate, all via ORCA (no xtb/GFN-FF/GFN2-xTB: confirmed
directly in this environment that both require the external `xtb` binary,
which isn't installed -- see rank_candidates_pm3.py's docstring):

  1. Reuse enumerate_protonation_candidates.py/rank_candidates_pm3.py's
     RDKit embed + MMFF/UFF pre-optimization, then an ORCA PM3 geometry
     optimization (rank_candidates_pm3.run_orca_pm3).
  2. Run a short ORCA PM3 Born-Oppenheimer MD trajectory (ORCA's built-in
     `%md` block, aka "SANdy" -- confirmed working in this environment
     without xtb; syntax below was reverse-engineered from the parser's own
     error messages and strings in the `orca_md` binary, since the two URLs
     ORCA prints for this module, sanscript.org/orcamd and
     brehm-research.de/sanscript, are both undocumented stub pages).
  3. Sample N frames from the production segment (after discarding an
     equilibration prefix), each already carrying its own PM3 potential
     energy from the trajectory dump.
  4. Run a fast ORCA PM3 single-point (Print[P_Mulliken]) on each sampled
     frame to get per-atom Mulliken charges -- MOBCAL needs a real charge
     distribution, not a formal-charge placeholder.
  5. Wrap each frame as an isicle.geometry.Geometry (RDKit mol + that
     frame's coordinates/charges/energy) and run MOBCAL (mobcal-shm,
     vendored at third_party/mobcal-shm/ -- see its PROVENANCE.md) via
     isicle.mobility.calculate_ccs, buffer gas nitrogen (this dataset is
     Bruker timsTOF/N2, not the isicle default helium).
  6. Boltzmann-weight (by relative PM3 electronic energy, i.e. no
     vibrational/entropic correction -- an approximation) the per-frame CCS
     values into one predicted CCS for the candidate.

Known approximations, flagged explicitly rather than silently accepted:

  - Bromine and iodine are NOT in MOBCAL's standard atomtype parameter table
    (neither isicle's bundled resources/lennard_jones.txt nor mobcal-shm's
    own bundled atomtype_parameters.in cover anything past Cl -- confirmed
    directly, this crashes with "type not defined for atom number N"). Br/I
    rows in --atomtype-params (default: resources/atomtype_parameters_extended.in,
    committed alongside this script) were DERIVED, not taken from a published
    source: pulled H/C/N/O/F/Cl/S/P/Si/Na/Fe UFF van-der-Waals parameters
    (x1, D1) from RDKit's own bundled UFF table
    (Code/ForceField/UFF/Params.cpp upstream), then transferred Br/I's UFF
    values onto MOBCAL's existing Cl row by ratio (same halogen group,
    adjacent period -- the periodic-table-neighbor with a real, fitted
    MOBCAL value): ROLJ_X = ROLJ_Cl * (x1_X / x1_Cl), EOLJ_X = EOLJ_Cl *
    (D1_X / D1_Cl). Spot-checked against two small Br-containing CCSBase
    drift-tube/N2 references (brompheniramine-like, dibromoaniline-like):
    predicted vs experimental CCS differed by +9% and -3% respectively (no
    consistent sign) -- reasonable for a single-conformer PM3 check with no
    MD averaging yet, not a large systematic error, but this is NOT a
    literature-validated parameter and should not be trusted beyond relative
    candidate-vs-candidate comparisons for the same molecule (which is all
    this script is actually used for).
  - Frame charges are Mulliken/PM3, and frame energies used for Boltzmann
    weighting are PM3 electronic energies only (no vibrational or entropic
    correction).

Cost (measured directly, 37-atom +1 candidate, single core): ORCA PM3 MD runs
at ~13.4 ps/hour. A --md-equil-steps 500 --md-prod-steps 2000 run (0.5 fs
timestep -> 1.25 ps total) is ~5-6 minutes per candidate, before the
per-frame Mulliken single-points and MOBCAL calls (seconds each, not the
bottleneck). This is a validation-subset tool (roadmap Step 5's ~30-100
molecules), not something to run over all ~102,000 secondary-role rows in
model2.parquet -- at this cost that would take months.

Usage (run under the GXS_torch pyenv environment, which has isicle
installed; rdkit/torch/pyarrow/scipy/tqdm are all already there):
    /home/rodvrees/.pyenv/versions/3.11.9/envs/GXS_torch/bin/python \\
        scripts/predict_candidate_ccs_md.py \\
        --input outputs/protonation_candidates/model2.parquet \\
        --output outputs/protonation_candidates/candidate_ccs_md.parquet \\
        --n-molecules 50 --n-jobs 8

    # Smoke test on a couple of candidates first:
    ... --limit 4 --n-jobs 2 --output outputs/protonation_candidates/candidate_ccs_md_smoke.parquet
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rank_candidates_pm3 import embed_and_optimize, mol_to_xyz_block  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DEFAULT_ORCA_BIN = "/home/rodvrees/orca_6_0_0/orca"
DEFAULT_MOBCAL_BIN = str(
    Path(__file__).resolve().parent.parent / "third_party" / "mobcal-shm" / "mobcal_shm"
)
DEFAULT_ATOMTYPE_PARAMS = str(
    Path(__file__).resolve().parent / "resources" / "atomtype_parameters_extended.in"
)

FINAL_ENERGY_RE = re.compile(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)")
FRAME_HEADER_RE = re.compile(
    r"ORCA AIMD Position Step (\d+), t=([\d.]+) fs, E_Pot=(-?[\d.]+) Hartree"
)
MULLIKEN_BLOCK_RE = re.compile(
    r"MULLIKEN ATOMIC CHARGES\s*\n-+\n(.*?)\nSum of atomic charges", re.S
)

HARTREE_TO_KCALMOL = 627.5094740631
KCAL_PER_MOL_PER_K = 0.0019872041  # gas constant R in kcal/(mol*K)
BUFFER_GAS_MASS = {"nitrogen": 28.0134, "helium": 4.0026}


def _orca_env(orca_bin: str) -> dict[str, str]:
    env = dict(os.environ)
    env["PATH"] = str(Path(orca_bin).parent) + os.pathsep + env.get("PATH", "")
    return env


def run_orca_pm3_md(
    xyz_block: str,
    charge: int,
    orca_bin: str,
    n_equil_steps: int,
    n_prod_steps: int,
    timestep_fs: float,
    temp_k: float,
    timeout_seconds: int,
    scratch_root: Optional[str],
) -> tuple[list[tuple[float, list[str], list[tuple[float, float, float]]]], str]:
    """Run an ORCA PM3 Born-Oppenheimer MD trajectory (ORCA's `%md`/SANdy
    module). Returns (frames, status) where frames is a list of
    (energy_hartree, elements, coords) for every dumped step (equilibration
    included -- the caller discards the equilibration prefix and subsamples).

    `dump position format xyz stride 1 filename "..." replace` is the
    correct keyword (confirmed by reverse-engineering: an initial guess of
    `Dump_Position` failed with "Unknown identifier", and the correct form
    was found via `strings` on the compiled `orca_md` binary itself, which
    embeds usage examples). `Run` takes an integer step count, not a time
    value (also confirmed by trial: `Run 20_fs` fails, `Run 40` works)."""
    n_total = n_equil_steps + n_prod_steps
    with tempfile.TemporaryDirectory(prefix="orca_md_", dir=scratch_root) as scratch:
        traj_name = "traj.xyz"
        inp_path = Path(scratch) / "job.inp"
        inp_path.write_text(
            f"! PM3\n"
            f"%md\n"
            f"   Timestep    {timestep_fs}_fs\n"
            f"   Initvel     {temp_k}_K\n"
            f"   Thermostat  Berendsen {temp_k}_K  timecon 100_fs\n"
            f"   dump position format xyz stride 1 filename \"{traj_name}\" replace\n"
            f"   Run  {n_total}\n"
            f"end\n"
            f"* xyz {charge} 1\n{xyz_block}\n*\n"
        )

        try:
            proc = subprocess.run(
                [orca_bin, str(inp_path)],
                cwd=scratch, capture_output=True, text=True,
                timeout=timeout_seconds, env=_orca_env(orca_bin),
            )
        except subprocess.TimeoutExpired:
            return [], "orca_md_timeout"

        if "ORCA TERMINATED NORMALLY" not in (proc.stdout + proc.stderr):
            return [], "orca_md_failed"

        traj_path = Path(scratch) / traj_name
        if not traj_path.exists():
            return [], "no_trajectory"

        frames = _parse_xyz_trajectory(traj_path.read_text())
        return frames, "ok" if frames else "empty_trajectory"


def _parse_xyz_trajectory(
    text: str,
) -> list[tuple[float, list[str], list[tuple[float, float, float]]]]:
    """Parse ORCA AIMD's multi-frame XYZ dump. Each frame: (energy_hartree,
    element_symbols, xyz_coords)."""
    frames = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or not line.isdigit():
            i += 1
            continue
        n_atoms = int(line)
        header = lines[i + 1]
        m = FRAME_HEADER_RE.search(header)
        if m is None:
            i += 1
            continue
        energy = float(m.group(3))
        elements = []
        coords = []
        for j in range(n_atoms):
            parts = lines[i + 2 + j].split()
            elements.append(parts[0])
            coords.append((float(parts[1]), float(parts[2]), float(parts[3])))
        frames.append((energy, elements, coords))
        i += 2 + n_atoms
    return frames


def run_orca_pm3_mulliken(
    xyz_block: str, charge: int, orca_bin: str, timeout_seconds: int, scratch_root: Optional[str]
) -> tuple[Optional[float], Optional[list[float]], str]:
    """Single-point PM3 (no optimization) with Mulliken population analysis.
    Returns (energy_hartree, charges, status)."""
    with tempfile.TemporaryDirectory(prefix="orca_sp_", dir=scratch_root) as scratch:
        inp_path = Path(scratch) / "job.inp"
        inp_path.write_text(
            f"! PM3\n%output\n  Print[P_Mulliken] 1\nend\n* xyz {charge} 1\n{xyz_block}\n*\n"
        )
        try:
            proc = subprocess.run(
                [orca_bin, str(inp_path)],
                cwd=scratch, capture_output=True, text=True,
                timeout=timeout_seconds, env=_orca_env(orca_bin),
            )
        except subprocess.TimeoutExpired:
            return None, None, "orca_sp_timeout"

        output = proc.stdout + proc.stderr
        energies = FINAL_ENERGY_RE.findall(output)
        if not energies:
            return None, None, "orca_sp_no_energy"

        block = MULLIKEN_BLOCK_RE.search(output)
        if block is None:
            return float(energies[-1]), None, "no_mulliken_block"

        charges = [float(line.split(":")[1].strip()) for line in block.group(1).strip().splitlines()]
        return float(energies[-1]), charges, "ok"


def _quench_frame(
    mol_template: Chem.Mol, coords: list[tuple[float, float, float]]
) -> list[tuple[float, float, float]]:
    """Raw MD snapshots can carry transient close contacts (unconstrained
    0.5 fs PM3 MD, explicit H's -- fine for sampling, not for feeding
    straight into a trajectory-method CCS calculator). A quick local MMFF
    (UFF fallback) minimization from the snapshot's own geometry relaxes
    those contacts while staying in the same conformational basin (it's a
    local, not global, minimization) -- standard practice before
    trajectory-method CCS calculations on MD-derived structures. (An
    initial run's every-frame MOBCAL hang looked like this at first, but
    turned out to be an unrelated concurrency bug in mobcal_shm itself --
    see isicle/mobility.py's MobcalWrapper.submit() -- not a geometry
    problem; this quench step is kept anyway since it's cheap and still
    good practice.)"""
    mol = Chem.Mol(mol_template)
    conf = mol.GetConformer()
    for i, (x, y, z) in enumerate(coords):
        conf.SetAtomPosition(i, Point3D(x, y, z))
    try:
        result = AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
    except Exception:  # noqa: BLE001 - MMFF params can be unavailable for some atom types
        result = -1
    if result != 0:
        try:
            AllChem.UFFOptimizeMolecule(mol, maxIters=200)
        except Exception:  # noqa: BLE001 - proceed with the raw snapshot as a last resort
            pass
    conf = mol.GetConformer()
    return [tuple(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())]


def _frame_geometry(mol_template: Chem.Mol, coords: list[tuple[float, float, float]], basename: str):
    """Copy of the candidate's RDKit mol (same connectivity/atom order as
    every other frame) with the conformer positions overwritten by this MD
    frame's coordinates."""
    import isicle.geometry

    mol = Chem.Mol(mol_template)
    conf = mol.GetConformer()
    for i, (x, y, z) in enumerate(coords):
        conf.SetAtomPosition(i, Point3D(x, y, z))
    return isicle.geometry.Geometry(mol=mol, basename=basename)


def _frame_ccs(
    mol_template: Chem.Mol,
    frame_idx: int,
    coords: list[tuple[float, float, float]],
    charges: list[float],
    energy_hartree: float,
    basename: str,
    mobcal_bin: str,
    atomtype_params: str,
    buffer_gas: str,
    mobcal_itn: int,
    mobcal_inp: int,
    mobcal_imp: int,
    mobcal_timeout_seconds: int,
) -> Optional[float]:
    import isicle.mobility

    geom = _frame_geometry(mol_template, coords, f"{basename}_f{frame_idx}")
    geom._energy = energy_hartree
    geom._charge = charges
    # Safe to run concurrently across --n-jobs: each call gets its own
    # temp_dir (fresh per Geometry/wrapper instance), and isicle's
    # MobcalWrapper.submit() runs mobcal_shm with cwd=temp_dir plus a
    # per-dir placeholder file, giving each invocation a distinct
    # shared-memory key (see that function's docstring for why this
    # matters -- it's the fix for a real, confirmed concurrency bug).
    wrapper = isicle.mobility.calculate_ccs(
        geom, buffer_gas=buffer_gas, buffer_gas_mass=BUFFER_GAS_MASS[buffer_gas],
        command=mobcal_bin, lennard_jones=atomtype_params,
        itn=mobcal_itn, inp=mobcal_inp, imp=mobcal_imp, processes=1,
        timeout=mobcal_timeout_seconds,
    )
    try:
        parsed = wrapper.parse()
    except Exception:
        # Malformed/absent .out (e.g. mobcal timed out mid-run, or crashed)
        return None
    ccs = parsed.get("ccs")
    return ccs["mean"] if ccs else None


def _set_mobcal_core_offset(block_size: int) -> None:
    """mobcal_shm pins its (forked) worker threads to CPU cores 0, 1, 2, ...
    on every invocation regardless of caller (see fork_and_bind.c, patched
    to read this) -- concurrent candidates would otherwise all fight over
    the same 1-2 physical cores no matter how many are available (confirmed
    directly: several concurrent calls all stalled at the identical point
    in their output). ProcessPoolExecutor assigns each worker process a
    stable, unique, sequential identity -- use it as a core-block index so
    concurrent workers land on disjoint cores. block_size must comfortably
    exceed the actual cores mobcal_shm's fork tree uses for the configured
    NUM_THREADS (confirmed empirically: a spacing of 2 was NOT enough
    headroom and still collided; a spacing of 5 was reliably clean)."""
    import multiprocessing

    identity = multiprocessing.current_process()._identity
    worker_index = identity[0] if identity else 0
    os.environ["MOBCAL_CORE_OFFSET"] = str(worker_index * block_size)


def predict_candidate_ccs(job: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    _set_mobcal_core_offset(args.mobcal_core_block_size)
    result = dict(job)
    smiles = job["candidate_smiles"]
    charge = job["charge_state"]
    basename = re.sub(r"[^A-Za-z0-9]", "_", f"{job['inchikey']}_{charge}")[:60]

    mol, embed_status = embed_and_optimize(smiles)
    if mol is None:
        result.update(predicted_ccs=None, predicted_ccs_std=None, n_frames_used=0, status=embed_status)
        return result

    actual_charge = Chem.GetFormalCharge(mol)
    if actual_charge != charge:
        result.update(predicted_ccs=None, predicted_ccs_std=None, n_frames_used=0,
                       status=f"charge_mismatch_{actual_charge}")
        return result

    xyz_block = mol_to_xyz_block(mol)
    frames, md_status = run_orca_pm3_md(
        xyz_block, actual_charge, args.orca_bin,
        args.md_equil_steps, args.md_prod_steps, args.md_timestep_fs, args.md_temp_k,
        args.md_timeout_seconds, args.scratch_dir,
    )
    if md_status != "ok":
        result.update(predicted_ccs=None, predicted_ccs_std=None, n_frames_used=0, status=md_status)
        return result

    production = frames[args.md_equil_steps:]
    if not production:
        result.update(predicted_ccs=None, predicted_ccs_std=None, n_frames_used=0, status="no_production_frames")
        return result
    sampled = production[:: max(1, len(production) // args.n_frames)][: args.n_frames]

    ccs_values = []
    energies_kcal = []
    for idx, (energy, elements, coords) in enumerate(sampled):
        coords = _quench_frame(mol, coords)
        frame_xyz = "\n".join(
            f"{el:2s} {x:12.6f} {y:12.6f} {z:12.6f}" for el, (x, y, z) in zip(elements, coords)
        )
        sp_energy, charges, sp_status = run_orca_pm3_mulliken(
            frame_xyz, actual_charge, args.orca_bin, args.md_timeout_seconds, args.scratch_dir
        )
        if sp_status != "ok" or charges is None:
            continue
        ccs = _frame_ccs(
            mol, idx, coords, charges, sp_energy, basename,
            args.mobcal_bin, args.atomtype_params, args.buffer_gas,
            args.mobcal_itn, args.mobcal_inp, args.mobcal_imp,
            args.mobcal_timeout_seconds,
        )
        if ccs is None:
            continue
        ccs_values.append(ccs)
        energies_kcal.append(sp_energy * HARTREE_TO_KCALMOL)

    if not ccs_values:
        result.update(predicted_ccs=None, predicted_ccs_std=None, n_frames_used=0, status="no_valid_frames")
        return result

    energies_kcal = np.array(energies_kcal)
    weights = np.exp(-(energies_kcal - energies_kcal.min()) / (KCAL_PER_MOL_PER_K * args.md_temp_k))
    weights /= weights.sum()
    ccs_values = np.array(ccs_values)
    weighted_mean = float(np.sum(weights * ccs_values))
    weighted_std = float(np.sqrt(np.sum(weights * (ccs_values - weighted_mean) ** 2)))

    result.update(
        predicted_ccs=weighted_mean, predicted_ccs_std=weighted_std,
        n_frames_used=len(ccs_values), status="ok",
    )
    return result


OUTPUT_SCHEMA = pa.schema(
    [
        ("inchikey", pa.string()),
        ("charge_state", pa.int8()),
        ("candidate_smiles", pa.string()),
        ("charged_atom_idx", pa.int32()),
        ("predicted_ccs", pa.float64()),
        ("predicted_ccs_std", pa.float64()),
        ("n_frames_used", pa.int32()),
        ("status", pa.string()),
    ]
)


def select_validation_candidates(model2_df: pd.DataFrame, n_molecules: Optional[int], seed: int) -> pd.DataFrame:
    """Multi-candidate groups only (nothing to validate for
    unambiguous_single_site groups); optionally a random subset of
    molecules, matching roadmap Step 5's ~30-100 molecule validation scope."""
    group_sizes = model2_df.groupby(["inchikey", "charge_state"]).size()
    multi_groups = group_sizes[group_sizes >= 2].index
    df = model2_df.set_index(["inchikey", "charge_state"]).loc[multi_groups].reset_index()
    df = df[df["match_confidence"] != "unambiguous_single_site"]

    if n_molecules is not None:
        keys = df[["inchikey", "charge_state"]].drop_duplicates()
        sampled_keys = keys.sample(n=min(n_molecules, len(keys)), random_state=seed)
        df = df.merge(sampled_keys, on=["inchikey", "charge_state"], how="inner")

    return df.drop_duplicates(subset=["inchikey", "charge_state", "candidate_smiles"])


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="build_protomer_labels.py's model2 output Parquet")
    parser.add_argument("--output", type=Path, required=True, help="Output Parquet for MD-predicted candidate CCS")
    parser.add_argument("--orca-bin", type=str, default=DEFAULT_ORCA_BIN)
    parser.add_argument("--mobcal-bin", type=str, default=DEFAULT_MOBCAL_BIN)
    parser.add_argument("--atomtype-params", type=str, default=DEFAULT_ATOMTYPE_PARAMS)
    parser.add_argument("--buffer-gas", choices=["nitrogen", "helium"], default="nitrogen",
                         help="This dataset is Bruker timsTOF/N2, not isicle's helium default")
    parser.add_argument("--n-molecules", type=int, default=50,
                         help="Random subset of (inchikey, charge_state) groups to validate (default: 50, matching roadmap Step 5's scope)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--md-equil-steps", type=int, default=500)
    parser.add_argument("--md-prod-steps", type=int, default=2000)
    parser.add_argument("--md-timestep-fs", type=float, default=0.5)
    parser.add_argument("--md-temp-k", type=float, default=300.0)
    parser.add_argument("--md-timeout-seconds", type=int, default=1800)
    parser.add_argument("--n-frames", type=int, default=10, help="Frames sampled from the production segment per candidate")
    parser.add_argument("--mobcal-itn", type=int, default=10)
    parser.add_argument("--mobcal-inp", type=int, default=8)
    parser.add_argument("--mobcal-imp", type=int, default=128)
    parser.add_argument("--mobcal-timeout-seconds", type=int, default=120,
                         help="Hard per-frame timeout for mobcal_shm, as a backstop; a timed-out "
                              "frame is simply dropped, not treated as a fatal error")
    parser.add_argument("--mobcal-core-block-size", type=int, default=5,
                         help="CPU cores reserved per concurrent worker for mobcal_shm's own "
                              "core-pinned fork tree (see _set_mobcal_core_offset) -- must exceed "
                              "--n-jobs * this value's worth of cores actually being available, "
                              "or concurrent workers will wrap around and collide on cores again")
    parser.add_argument("--n-jobs", type=int, default=min(8, os.cpu_count() or 1),
                         help="Concurrent candidates (each MD run is single-core, but reserve "
                              "--n-jobs * --mobcal-core-block-size cores in total, not just "
                              "--n-jobs, since each worker's mobcal_shm call also pins cores)")
    parser.add_argument("--chunk-size", type=int, default=20)
    parser.add_argument("--limit", type=int, default=None, help="Cap on number of candidates (smoke testing)")
    parser.add_argument("--scratch-dir", type=str, default="/dev/shm" if Path("/dev/shm").is_dir() else None)
    args = parser.parse_args(argv)

    if not Path(args.orca_bin).exists():
        raise FileNotFoundError(f"ORCA binary not found at {args.orca_bin}")
    if not Path(args.mobcal_bin).exists():
        raise FileNotFoundError(
            f"mobcal_shm binary not found at {args.mobcal_bin} -- build it first: "
            f"see third_party/mobcal-shm/PROVENANCE.md"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    model2_df = pd.read_parquet(args.input)
    candidates_df = select_validation_candidates(model2_df, args.n_molecules, args.seed)
    if args.limit is not None:
        candidates_df = candidates_df.head(args.limit)

    jobs = candidates_df[
        ["inchikey", "charge_state", "candidate_smiles", "charged_atom_idx"]
    ].to_dict("records")
    logger.info(
        "Predicting MD/MOBCAL CCS for %d candidates across %d molecule groups (%d parallel jobs)",
        len(jobs), candidates_df[["inchikey", "charge_state"]].drop_duplicates().shape[0], args.n_jobs,
    )

    stats: dict[str, int] = {}
    writer = pq.ParquetWriter(args.output, OUTPUT_SCHEMA)
    buffer: list[dict[str, Any]] = []

    def flush() -> None:
        if not buffer:
            return
        table = pa.Table.from_pandas(pd.DataFrame(buffer), schema=OUTPUT_SCHEMA, preserve_index=False)
        writer.write_table(table)
        buffer.clear()

    try:
        with ProcessPoolExecutor(max_workers=args.n_jobs) as executor:
            futures = [executor.submit(predict_candidate_ccs, job, args) for job in jobs]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Predicting candidate CCS (ORCA MD + MOBCAL)"):
                result = future.result()
                buffer.append(result)
                stats[result["status"]] = stats.get(result["status"], 0) + 1
                if len(buffer) >= args.chunk_size:
                    flush()
        flush()
    finally:
        writer.close()

    logger.info("Run summary:")
    for key in sorted(stats):
        logger.info("  %s: %d", key, stats[key])
    logger.info("Wrote MD-predicted candidate CCS to %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
