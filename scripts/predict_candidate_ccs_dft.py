#!/usr/bin/env python3
"""
Reassessing the level of theory behind Step 2's candidate ranking/matching,
per the discussion in matching_validation_breakdown.ipynb (sections 6 and 9):
neither the positional matching heuristic nor its MD/MOBCAL-based validation
showed any signal, and a ~5x longer MD trajectory ruled out "insufficient
sampling" as the explanation. The two things that could still be the actual
bottleneck, both currently done at PM3 (a decades-old semi-empirical method
not known for reliable relative protonation-site energetics):

  1. Candidate ranking: the "which site is more stable" ranking that
     determines the positional heuristic's "rank 1" pick. The candidates
     currently on disk in model2.parquet are ranked by `pka_mean`
     (Dimorphite-DL's tabulated literature pKaH lookup by SMARTS site type
     -- a fixed per-site-type value, not even molecule-specific), which is a
     cruder proxy than PM3 energy, not a better one. Neither has been
     checked against a real electronic-structure calculation.
  2. Charges feeding MOBCAL's potential: currently PM3 Mulliken, which
     drives the ion-induced-dipole/quadrupole terms MOBCAL's trajectory
     method actually integrates over.

This script re-does both, for a single, deliberately CHEAP escalation: reuse
the same PM3-optimized geometry already used everywhere else in this
pipeline (RDKit embed/MMFF + ORCA PM3 Opt, rank_candidates_pm3.py's
functions), and layer ONE r2SCAN-3c single point on top -- no new MD, no
DFT-level dynamics. r2SCAN-3c is a modern composite DFT method (small basis,
built-in dispersion/BSSE correction) chosen specifically because it's
dramatically better than PM3 for both relative energetics and charges while
still being single-point-cheap (confirmed directly: ~38s for a 37-atom
candidate on this machine, vs ~2.5s for the equivalent PM3 single point --
more expensive, but nothing like redoing MD at DFT level would cost).

Per candidate:
  1. Reuse embed_and_optimize + run_orca_pm3 (rank_candidates_pm3.py) to get
     the same PM3-optimized geometry already used throughout this pipeline.
  2. Run ONE ORCA r2SCAN-3c single point on that geometry with
     Print[P_Mulliken], giving a DFT energy (for re-ranking) and DFT-level
     Mulliken charges in one calculation.
  3. Feed that geometry + DFT charges to MOBCAL (mobcal-shm, same as
     predict_candidate_ccs_md.py) for a single (non-MD-averaged) predicted
     CCS -- this isolates "do better charges on the SAME geometry change
     anything" from "does more conformational sampling matter" (already
     answered: no, see the notebook's longer-MD section).

This is a pilot on a small, deliberately clean subset (Model 1 `positive` +
exactly 2 candidates -- the same 34-group/68-candidate slice already used
for the notebook's cleanest binomial test), not a re-run of the full
validation set. Scale up only if this shows something worth the extra cost.

Usage:
    /home/rodvrees/.pyenv/versions/3.11.9/envs/GXS_torch/bin/python \\
        scripts/predict_candidate_ccs_dft.py \\
        --candidates /tmp/pilot_candidates.parquet \\
        --output outputs/protonation_candidates/candidate_ccs_dft_pilot.parquet \\
        --n-jobs 8
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

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rank_candidates_pm3 import embed_and_optimize, mol_to_xyz_block  # noqa: E402
from predict_candidate_ccs_md import (  # noqa: E402
    DEFAULT_MOBCAL_BIN,
    DEFAULT_ATOMTYPE_PARAMS,
    BUFFER_GAS_MASS,
    MULLIKEN_BLOCK_RE,
    FINAL_ENERGY_RE,
    _orca_env,
    _frame_geometry,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DEFAULT_ORCA_BIN = "/home/rodvrees/orca_6_0_0/orca"


def run_orca_singlepoint_mulliken(
    xyz_block: str, charge: int, method: str, orca_bin: str, timeout_seconds: int, scratch_root: Optional[str],
) -> tuple[Optional[float], Optional[list[float]], str]:
    """Single-point (no optimization) at the given ORCA method keyword, with
    Mulliken population analysis. Same parsing as
    predict_candidate_ccs_md.run_orca_pm3_mulliken, generalized to an
    arbitrary method string so it can run r2SCAN-3c instead of PM3."""
    with tempfile.TemporaryDirectory(prefix="orca_dft_sp_", dir=scratch_root) as scratch:
        inp_path = Path(scratch) / "job.inp"
        inp_path.write_text(
            f"! {method}\n%output\n  Print[P_Mulliken] 1\nend\n* xyz {charge} 1\n{xyz_block}\n*\n"
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


def _static_ccs(
    mol: Chem.Mol, charges: list[float], energy_hartree: float, basename: str,
    mobcal_bin: str, atomtype_params: str, buffer_gas: str,
    mobcal_itn: int, mobcal_inp: int, mobcal_imp: int, mobcal_threads: int, mobcal_timeout_seconds: int,
) -> Optional[float]:
    import isicle.mobility

    conf = mol.GetConformer()
    coords = [tuple(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())]
    geom = _frame_geometry(mol, coords, basename)
    geom._energy = energy_hartree
    geom._charge = charges
    wrapper = isicle.mobility.calculate_ccs(
        geom, buffer_gas=buffer_gas, buffer_gas_mass=BUFFER_GAS_MASS[buffer_gas],
        command=mobcal_bin, lennard_jones=atomtype_params,
        itn=mobcal_itn, inp=mobcal_inp, imp=mobcal_imp, processes=mobcal_threads,
        timeout=mobcal_timeout_seconds,
    )
    try:
        parsed = wrapper.parse()
    except Exception:
        return None
    ccs = parsed.get("ccs")
    return ccs["mean"] if ccs else None


def process_candidate(job: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    result = dict(job)
    smiles = job["candidate_smiles"]
    charge = job["charge_state"]
    basename = re.sub(r"[^A-Za-z0-9]", "_", f"{job['inchikey']}_{charge}")[:60]

    mol, embed_status = embed_and_optimize(smiles)
    if mol is None:
        result.update(dft_energy_hartree=None, dft_ccs=None, status=embed_status)
        return result

    actual_charge = Chem.GetFormalCharge(mol)
    if actual_charge != charge:
        result.update(dft_energy_hartree=None, dft_ccs=None, status=f"charge_mismatch_{actual_charge}")
        return result

    xyz_block = mol_to_xyz_block(mol)

    # Step 1: same PM3 geometry optimization used everywhere else in this
    # pipeline -- NOT re-deriving a new geometry, just reproducing the
    # existing one so this is a clean apples-to-apples "same geometry,
    # better electronic structure" test. rank_candidates_pm3.run_orca_pm3
    # only returns the final energy/status, not the geometry, so this reruns
    # the PM3 Opt itself and parses both from the SAME output (NOT two
    # separate calls -- and NOT `re.search`'s first match, which would grab
    # the *initial*, not converged, geometry: "CARTESIAN COORDINATES
    # (ANGSTROEM)" is printed once per optimization step, confirmed directly
    # -- 13 occurrences for a typical run here -- so this takes the LAST one).
    with tempfile.TemporaryDirectory(prefix="orca_pm3_opt_", dir=args.scratch_dir) as scratch:
        inp_path = Path(scratch) / "job.inp"
        inp_path.write_text(f"! PM3 Opt\n* xyz {actual_charge} 1\n{xyz_block}\n*\n")
        try:
            proc = subprocess.run(
                [args.orca_bin, str(inp_path)], cwd=scratch, capture_output=True, text=True,
                timeout=args.pm3_timeout_seconds, env=_orca_env(args.orca_bin),
            )
        except subprocess.TimeoutExpired:
            result.update(pm3_energy_hartree=None, dft_energy_hartree=None, dft_ccs=None, status="pm3_opt_timeout")
            return result
        output = proc.stdout + proc.stderr

        energies = FINAL_ENERGY_RE.findall(output)
        if not energies:
            result.update(pm3_energy_hartree=None, dft_energy_hartree=None, dft_ccs=None, status="pm3_opt_no_energy")
            return result
        pm3_energy = float(energies[-1])

        geom_blocks = re.findall(r"CARTESIAN COORDINATES \(ANGSTROEM\)\s*\n-+\n(.*?)\n\n", output, re.S)
        if not geom_blocks:
            result.update(pm3_energy_hartree=pm3_energy, dft_energy_hartree=None, dft_ccs=None,
                           status="pm3_opt_no_geometry")
            return result
        conf = mol.GetConformer()
        for i, line in enumerate(geom_blocks[-1].strip().splitlines()):
            parts = line.split()
            conf.SetAtomPosition(i, (float(parts[1]), float(parts[2]), float(parts[3])))
        opt_xyz_block = mol_to_xyz_block(mol)

    # Step 2: ONE r2SCAN-3c single point on that same geometry.
    dft_energy, charges, sp_status = run_orca_singlepoint_mulliken(
        opt_xyz_block, actual_charge, args.dft_method, args.orca_bin, args.dft_timeout_seconds, args.scratch_dir,
    )
    if sp_status != "ok" or charges is None:
        result.update(dft_energy_hartree=dft_energy, dft_ccs=None, status=f"dft_sp_{sp_status}")
        return result

    # Step 3: MOBCAL on this single geometry with the DFT charges.
    ccs = _static_ccs(
        mol, charges, dft_energy, basename,
        args.mobcal_bin, args.atomtype_params, args.buffer_gas,
        args.mobcal_itn, args.mobcal_inp, args.mobcal_imp, args.mobcal_threads, args.mobcal_timeout_seconds,
    )

    result.update(
        pm3_energy_hartree=pm3_energy,
        dft_energy_hartree=dft_energy,
        dft_ccs=ccs,
        status="ok" if ccs is not None else "mobcal_failed",
    )
    return result


OUTPUT_SCHEMA = pa.schema(
    [
        ("inchikey", pa.string()),
        ("charge_state", pa.int8()),
        ("candidate_smiles", pa.string()),
        ("charged_atom_idx", pa.int32()),
        ("observed_ccs", pa.float32()),
        ("pka_mean", pa.float32()),
        ("pka_rank", pa.int32()),
        ("pm3_energy_hartree", pa.float64()),
        ("dft_energy_hartree", pa.float64()),
        ("dft_ccs", pa.float64()),
        ("status", pa.string()),
    ]
)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidates", type=Path, required=True,
                         help="Parquet with inchikey, charge_state, candidate_smiles, charged_atom_idx, "
                              "observed_ccs, pka_mean, pka_rank (the pilot subset to re-rank/re-predict)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--orca-bin", type=str, default=DEFAULT_ORCA_BIN)
    parser.add_argument("--mobcal-bin", type=str, default=DEFAULT_MOBCAL_BIN)
    parser.add_argument("--atomtype-params", type=str, default=DEFAULT_ATOMTYPE_PARAMS)
    parser.add_argument("--buffer-gas", choices=["nitrogen", "helium"], default="nitrogen")
    parser.add_argument("--dft-method", type=str, default="r2SCAN-3c")
    parser.add_argument("--pm3-timeout-seconds", type=int, default=300)
    parser.add_argument("--dft-timeout-seconds", type=int, default=600)
    parser.add_argument("--mobcal-itn", type=int, default=10)
    parser.add_argument("--mobcal-inp", type=int, default=8)
    parser.add_argument("--mobcal-imp", type=int, default=128)
    parser.add_argument("--mobcal-threads", type=int, default=4,
                         help="Threads per mobcal_shm call. Kept modest (not 32, unlike "
                              "predict_candidate_ccs_md.py's default) since --n-jobs candidates run "
                              "genuinely concurrently here (no shared gate, no MD to dominate the "
                              "timeline) -- --n-jobs * this must stay within the core count")
    parser.add_argument("--mobcal-timeout-seconds", type=int, default=300)
    parser.add_argument("--n-jobs", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--scratch-dir", type=str, default="/dev/shm" if Path("/dev/shm").is_dir() else None)
    args = parser.parse_args(argv)

    if not Path(args.orca_bin).exists():
        raise FileNotFoundError(f"ORCA binary not found at {args.orca_bin}")
    if not Path(args.mobcal_bin).exists():
        raise FileNotFoundError(f"mobcal_shm binary not found at {args.mobcal_bin}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    candidates_df = pd.read_parquet(args.candidates)

    # Tolerate work-lists (screen_protomer_candidates.py output) that lack the
    # observed_ccs / pka_rank columns the pilot input had: observed_ccs is
    # carried through only for the later matching step (not needed to compute a
    # predicted CCS here), and pka_rank is derived from pka_mean within each
    # group if absent. Both are in OUTPUT_SCHEMA, so they must exist as columns.
    if "observed_ccs" not in candidates_df.columns:
        candidates_df["observed_ccs"] = float("nan")
    if "pka_rank" not in candidates_df.columns:
        candidates_df["pka_rank"] = (
            candidates_df.groupby(["inchikey", "charge_state"])["pka_mean"]
            .rank(method="first", ascending=False)
            .fillna(0)
            .astype(int)
        )

    if args.limit is not None:
        candidates_df = candidates_df.head(args.limit)

    jobs = candidates_df.to_dict("records")
    logger.info(
        "Re-ranking + DFT-charge CCS for %d candidates across %d groups (%s, %d parallel jobs)",
        len(jobs), candidates_df[["inchikey", "charge_state"]].drop_duplicates().shape[0],
        args.dft_method, args.n_jobs,
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
            futures = [executor.submit(process_candidate, job, args) for job in jobs]
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"{args.dft_method} re-ranking + CCS"):
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
    logger.info("Wrote DFT re-ranking/CCS results to %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
