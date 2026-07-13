#!/usr/bin/env python3
"""
Upgrade for protomer_ccs_predictor_roadmap.md Step 2's candidate-ranking proxy
(a): "rank candidates by a fast semi-empirical proton affinity estimate (e.g.,
GFN2-xTB)". GFN2-xTB itself is not usable in this environment -- no standalone
`xtb` binary is installed (no conda/mamba to get it from conda-forge, no
matching pip wheel), and ORCA's own `! XTB2` keyword is just a wrapper around
that same missing binary (confirmed directly: ORCA aborts with "otool_xtb not
available. Trying xtb instead ... failed!"). ORCA 6.0.0 IS installed here, so
this script uses ORCA's native PM3 semi-empirical method instead, which needs
no external xtb binary.

For each charge-positioned candidate SMILES from Step 1
(scripts/enumerate_protonation_candidates.py):
  1. RDKit: add explicit Hs, embed a 3D conformer (ETKDGv3), pre-optimize with
     MMFF94 (falls back to UFF if MMFF parameters are unavailable for an atom
     type -- happens occasionally for less common elements/charge states).
  2. Write an ORCA input (`! PM3 Opt`, charge = the candidate's formal charge,
     multiplicity = 1) and run it via subprocess in an isolated scratch
     directory (PM3/NDO methods cannot run in ORCA's parallel mode in this
     build -- confirmed directly -- so each job is single-core; parallelism
     comes from running many independent jobs concurrently via --n-jobs).
  3. Parse the last "FINAL SINGLE POINT ENERGY" line as this candidate's PM3
     energy (Hartree).

All candidates for the same (inchikey, charge_state) share the same molecular
formula and total charge (same parent, one site protonated/deprotonated), so
their PM3 energies are directly comparable: the lowest-energy candidate is the
most stable protonation/deprotonation site, i.e. the proxy for "highest
effective proton affinity" build_protomer_labels.py needs for ranking.

This is much more expensive than Step 1's free Dimorphite-DL pKa lookup (a
real geometry optimization per candidate, ~1-10+ seconds each depending on
size) -- run it as a separate, resumable batch job, not inline in Step 1 or 2.

Usage:
    python scripts/rank_candidates_pm3.py \\
        --input outputs/protonation_candidates.parquet \\
        --output outputs/pm3_energies.parquet \\
        --n-jobs 16

    # Smoke test on a handful of candidates first:
    python scripts/rank_candidates_pm3.py --limit 20 \\
        --input outputs/protonation_candidates.parquet \\
        --output outputs/pm3_energies_smoke.parquet
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
from rdkit.Chem import AllChem
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DEFAULT_ORCA_BIN = "/home/rodvrees/orca_6_0_0/orca"
FINAL_ENERGY_RE = re.compile(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)")


def embed_and_optimize(smiles: str) -> tuple[Optional[Chem.Mol], str]:
    """3D embed + cheap force-field pre-optimization. Returns (mol, status);
    mol is None on failure."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, "parse_failed"
    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = 0xF00D
    cid = AllChem.EmbedMolecule(mol, params)
    if cid < 0:
        params.useRandomCoords = True
        cid = AllChem.EmbedMolecule(mol, params)
    if cid < 0:
        return None, "embed_failed"

    try:
        mmff_result = AllChem.MMFFOptimizeMolecule(mol, maxIters=1000)
    except Exception:  # noqa: BLE001 - MMFF params can be unavailable for some atom types
        mmff_result = -1
    if mmff_result != 0:
        try:
            AllChem.UFFOptimizeMolecule(mol, maxIters=1000)
        except Exception:  # noqa: BLE001 - proceed with the raw embed as a last resort
            pass

    return mol, "ok"


def mol_to_xyz_block(mol: Chem.Mol) -> str:
    conf = mol.GetConformer()
    lines = []
    for atom in mol.GetAtoms():
        p = conf.GetAtomPosition(atom.GetIdx())
        lines.append(f"{atom.GetSymbol():2s} {p.x:12.6f} {p.y:12.6f} {p.z:12.6f}")
    return "\n".join(lines)


def run_orca_pm3(xyz_block: str, charge: int, orca_bin: str, timeout_seconds: int) -> tuple[Optional[float], str]:
    """Run a single ORCA PM3 geometry optimization. Returns (energy_hartree,
    status); energy is None on failure. No %pal block: PM3/NDO methods can't
    run in ORCA's parallel mode in this build (confirmed directly)."""
    with tempfile.TemporaryDirectory(prefix="orca_pm3_") as scratch:
        inp_path = Path(scratch) / "job.inp"
        inp_path.write_text(f"! PM3 Opt\n* xyz {charge} 1\n{xyz_block}\n*\n")

        env = dict(os.environ)
        orca_dir = str(Path(orca_bin).parent)
        env["PATH"] = orca_dir + os.pathsep + env.get("PATH", "")

        try:
            proc = subprocess.run(
                [orca_bin, str(inp_path)],
                cwd=scratch, capture_output=True, text=True,
                timeout=timeout_seconds, env=env,
            )
        except subprocess.TimeoutExpired:
            return None, "orca_timeout"

        output = proc.stdout + proc.stderr
        energies = FINAL_ENERGY_RE.findall(output)
        if not energies:
            return None, "orca_no_energy"

        converged = "THE OPTIMIZATION HAS CONVERGED" in output
        return float(energies[-1]), "ok" if converged else "not_converged"


def compute_pm3_energy(job: dict[str, Any], orca_bin: str, timeout_seconds: int) -> dict[str, Any]:
    result = dict(job)
    mol, status = embed_and_optimize(job["candidate_smiles"])
    if mol is None:
        result.update(pm3_energy_hartree=None, pm3_status=status)
        return result

    actual_charge = Chem.GetFormalCharge(mol)
    if actual_charge != job["charge_state"]:
        # Shouldn't happen -- Step 1 already validated this -- but don't
        # silently run ORCA on a structure with the wrong charge if it does.
        result.update(pm3_energy_hartree=None, pm3_status=f"charge_mismatch_{actual_charge}")
        return result

    xyz_block = mol_to_xyz_block(mol)
    energy, orca_status = run_orca_pm3(xyz_block, actual_charge, orca_bin, timeout_seconds)
    result.update(pm3_energy_hartree=energy, pm3_status=orca_status)
    return result


OUTPUT_SCHEMA = pa.schema(
    [
        ("inchikey", pa.string()),
        ("charge_state", pa.int8()),
        ("candidate_smiles", pa.string()),
        ("charged_atom_idx", pa.int32()),
        ("pm3_energy_hartree", pa.float64()),
        ("pm3_status", pa.string()),
    ]
)


def run_ranking(
    candidates_df: pd.DataFrame, output_path: Path, orca_bin: str, timeout_seconds: int, n_jobs: int, chunk_size: int
) -> dict[str, int]:
    jobs = candidates_df[
        ["inchikey", "charge_state", "candidate_smiles", "charged_atom_idx"]
    ].to_dict("records")

    stats: dict[str, int] = {}
    writer = pq.ParquetWriter(output_path, OUTPUT_SCHEMA)
    buffer: list[dict[str, Any]] = []

    def flush() -> None:
        if not buffer:
            return
        table = pa.Table.from_pandas(pd.DataFrame(buffer), schema=OUTPUT_SCHEMA, preserve_index=False)
        writer.write_table(table)
        buffer.clear()

    try:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = [executor.submit(compute_pm3_energy, job, orca_bin, timeout_seconds) for job in jobs]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Running ORCA PM3"):
                result = future.result()
                buffer.append(result)
                stats[result["pm3_status"]] = stats.get(result["pm3_status"], 0) + 1
                if len(buffer) >= chunk_size:
                    flush()
        flush()
    finally:
        writer.close()

    return stats


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="Step 1's output Parquet")
    parser.add_argument("--output", type=Path, required=True, help="Output Parquet for PM3 energies")
    parser.add_argument("--orca-bin", type=str, default=DEFAULT_ORCA_BIN, help=f"Path to the orca binary (default: {DEFAULT_ORCA_BIN})")
    parser.add_argument("--timeout-seconds", type=int, default=300, help="Per-candidate ORCA timeout (default: 300)")
    parser.add_argument(
        "--n-jobs", type=int, default=min(16, os.cpu_count() or 1),
        help="Number of concurrent ORCA jobs (each is single-core; default: min(16, cpu_count))",
    )
    parser.add_argument("--chunk-size", type=int, default=200, help="Rows to buffer before writing to Parquet (default: 200)")
    parser.add_argument("--limit", type=int, default=None, help="Cap on number of candidates to process (for smoke testing)")
    args = parser.parse_args(argv)

    if not Path(args.orca_bin).exists():
        raise FileNotFoundError(f"ORCA binary not found at {args.orca_bin}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    candidates_df = pd.read_parquet(args.input)
    candidates_df = candidates_df[candidates_df["candidate_smiles"].notna()]
    if args.limit is not None:
        candidates_df = candidates_df.head(args.limit)
    logger.info("Ranking %d candidates via ORCA PM3 (%d parallel jobs)", len(candidates_df), args.n_jobs)

    stats = run_ranking(candidates_df, args.output, args.orca_bin, args.timeout_seconds, args.n_jobs, args.chunk_size)

    logger.info("Run summary:")
    for key in sorted(stats):
        logger.info("  %s: %d", key, stats[key])
    logger.info("Wrote PM3 energies to %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
