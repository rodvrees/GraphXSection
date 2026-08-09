"""Rank protonation/deprotonation microstates (protomers) of each molecule by
relative gas-phase energy, to assign abundance (-> intensity order -> observed
CCS peak) for protomer-aware training labels.

Concept (see ROADMAP.md "CHOSEN DIRECTION"): the observed IM peaks are
intensity-ordered and intensity tracks abundance; abundance is governed by
protonation energetics; so the LOWEST-energy charged microstate is the most
abundant = base peak. For [M-H]- the mirror holds (lowest-energy anion = most
acidic site = most abundant deprotomer). We never compute CCS: the observed CCS
is the label, energy only decides the assignment.

For microstates of ONE molecule the neutral energy cancels, so we compute only
the charged species and rank by their energy (the archived
proton_affinity_calculator.py computed the neutral too -- pure overhead).

Two-stage for efficiency (all ~800k candidates would be far too many DFT jobs):
  Stage 1 (pm3):  PM3-opt every candidate (~2.5 s), store energy + optimized xyz.
  Stage 2 (dft):  for MOLECULES whose ranking is uncertain at PM3 (near-tie),
                  or that carry a gap (clean secondary peak), or are anions
                  (PM3 is unreliable for anions), run an r2SCAN-3c single point
                  on the stored PM3 geometry -- for ALL candidates of that
                  molecule (never mix PM3 and DFT energies within one ranking).
  Stage 3 (rank): homogeneous final energy per molecule (all-DFT or all-PM3),
                  relative energies, Boltzmann populations, and per-molecule
                  protomer-vs-conformer signals.

Anions ([M-H]-) need diffuse basis functions; r2SCAN-3c's def2-mTZVPP has none,
so --dft-method-neg defaults to a minimally-augmented level. Confirm it runs in
your ORCA build (the smoke test does).

Reuses the lightweight rdkit helpers from rank_candidates_pm3 (embed_and_optimize,
mol_to_xyz_block); the ORCA callers are inlined here to avoid the isicle/MOBCAL
import chain (we do not compute CCS).

Starts from the ORIGINAL Enveda data (data/Enveda-180/enveda-180.jsonl.gz). The
protonation-candidate enumeration is not reimplemented here: this script calls
the validated pipeline in enumerate_protonation_candidates.py (load_spectra ->
build_secondary_species_table -> build_molecule_table -> run_candidate_generation,
the exact chain that produced outputs/protonation_candidates/enveda_widened.parquet)
and caches the result to --candidates. If that cache already exists it is reused
as-is, so pointing --candidates at an existing candidate parquet skips enumeration.
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
from rdkit import Chem, RDLogger

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rank_candidates_pm3 import embed_and_optimize, mol_to_xyz_block, DEFAULT_ORCA_BIN, FINAL_ENERGY_RE

RDLogger.DisableLog("rdApp.*")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

HARTREE_KCAL = 627.5095
KT_298_KCAL = 0.5924            # k_B T at 298.15 K in kcal/mol
_CART_RE = re.compile(r"CARTESIAN COORDINATES \(ANGSTROEM\)\s*\n-+\n(.*?)\n\n", re.S)
_SCF_FAIL = "SCF NOT CONVERGED AFTER"


def _orca_env(orca_bin: str) -> dict:
    env = dict(os.environ)
    env["PATH"] = str(Path(orca_bin).parent) + os.pathsep + env.get("PATH", "")
    return env


def _site_element(site_name: str) -> str:
    tag = str(site_name).split(",")[-1].strip()
    if tag.endswith("_N"):
        return "N"
    if tag.endswith("_O"):
        return "O"
    if tag.endswith("_S"):
        return "S"
    t = tag.lower()
    if any(k in t for k in ("amine", "aniline", "amide", "nitrogen", "pyridine", "imid", "azo", "nnh")):
        return "N"
    if any(k in t for k in ("hydroxyl", "carboxyl", "carbonyl", "ether", "alcohol", "oxygen")):
        return "O"
    if "sulf" in t or "thio" in t:
        return "S"
    return "?"


# --------------------------------------------------------------------------- #
# ORCA callers (inlined; no isicle dependency)
# --------------------------------------------------------------------------- #
def run_pm3_opt(xyz_block: str, charge: int, orca_bin: str, timeout_s: int,
                scratch_root: Optional[str]) -> tuple[Optional[float], Optional[str], str]:
    """PM3 geometry optimisation. Returns (energy_hartree, optimized_xyz_block,
    status). The optimized geometry (last CARTESIAN COORDINATES block) is kept so
    Stage 2 can run the DFT single point without re-optimising."""
    with tempfile.TemporaryDirectory(prefix="orca_pm3_", dir=scratch_root,
                                     ignore_cleanup_errors=True) as scratch:
        inp = Path(scratch) / "job.inp"
        inp.write_text(f"! PM3 Opt\n* xyz {charge} 1\n{xyz_block}\n*\n")
        try:
            proc = subprocess.run([orca_bin, str(inp)], cwd=scratch, capture_output=True,
                                  text=True, timeout=timeout_s, env=_orca_env(orca_bin))
        except subprocess.TimeoutExpired:
            return None, None, "pm3_timeout"
        out = proc.stdout + proc.stderr
        energies = FINAL_ENERGY_RE.findall(out)
        if not energies:
            return None, None, "pm3_no_energy"
        geom = _CART_RE.findall(out)
        opt_xyz = "\n".join(l.rstrip() for l in geom[-1].strip().splitlines()) if geom else None
        status = "ok" if "THE OPTIMIZATION HAS CONVERGED" in out else "pm3_not_converged"
        return float(energies[-1]), opt_xyz, status


def run_dft_sp(xyz_block: str, charge: int, method: str, orca_bin: str, timeout_s: int,
               scratch_root: Optional[str]) -> tuple[Optional[float], str]:
    """Single point at `method`. Returns (energy_hartree, status). Flags SCF
    non-convergence explicitly (important for anions)."""
    with tempfile.TemporaryDirectory(prefix="orca_dft_", dir=scratch_root,
                                     ignore_cleanup_errors=True) as scratch:
        inp = Path(scratch) / "job.inp"
        inp.write_text(f"! {method}\n* xyz {charge} 1\n{xyz_block}\n*\n")
        try:
            proc = subprocess.run([orca_bin, str(inp)], cwd=scratch, capture_output=True,
                                  text=True, timeout=timeout_s, env=_orca_env(orca_bin))
        except subprocess.TimeoutExpired:
            return None, "dft_timeout"
        out = proc.stdout + proc.stderr
        if _SCF_FAIL in out:
            return None, "dft_scf_not_converged"
        energies = FINAL_ENERGY_RE.findall(out)
        if not energies:
            return None, "dft_no_energy"
        return float(energies[-1]), "ok"


# --------------------------------------------------------------------------- #
# Stage 1: PM3 over all candidates
# --------------------------------------------------------------------------- #
_PM3_SCHEMA = pa.schema([
    ("inchikey", pa.string()), ("charge_state", pa.int8()),
    ("candidate_smiles", pa.string()), ("charged_atom_idx", pa.int32()),
    ("pm3_energy_hartree", pa.float64()), ("opt_xyz", pa.string()), ("pm3_status", pa.string()),
])


def _pm3_one(job: dict, orca_bin: str, timeout_s: int, scratch_root: Optional[str]) -> dict:
    r = {k: job[k] for k in ("inchikey", "charge_state", "candidate_smiles", "charged_atom_idx")}
    try:
        mol, st = embed_and_optimize(job["candidate_smiles"])
        if mol is None:
            return {**r, "pm3_energy_hartree": None, "opt_xyz": None, "pm3_status": st}
        if Chem.GetFormalCharge(mol) != job["charge_state"]:
            return {**r, "pm3_energy_hartree": None, "opt_xyz": None, "pm3_status": "charge_mismatch"}
        e, opt_xyz, st = run_pm3_opt(mol_to_xyz_block(mol), int(job["charge_state"]), orca_bin, timeout_s, scratch_root)
        return {**r, "pm3_energy_hartree": e, "opt_xyz": opt_xyz, "pm3_status": st}
    except Exception as exc:
        # A single job must never take down the whole pool (e.g. a temp-dir
        # cleanup race raising from inside run_pm3_opt) -- record and move on.
        return {**r, "pm3_energy_hartree": None, "opt_xyz": None,
                "pm3_status": f"worker_exception:{type(exc).__name__}"}


def _done_keys(path: Path, cols: list[str]) -> set:
    if not path.exists():
        return set()
    df = pd.read_parquet(path, columns=cols)
    return set(map(tuple, df[cols].itertuples(index=False, name=None)))


def _reconcile_partial(out_path: Path, key_cols: list[str]) -> None:
    """Merge a `<out_path>.partial.parquet` left behind by a run that crashed
    before `_run_pool` reached its own end-of-run merge. Without this, those
    already-computed rows are invisible to `_done_keys` and get silently
    recomputed (losing the compute the crashed run already paid for)."""
    partial = out_path.with_suffix(".partial.parquet")
    if not partial.exists():
        return
    logger.warning("Found stale %s from an interrupted run; merging into %s before resuming", partial, out_path)
    parts = [pd.read_parquet(partial)]
    if out_path.exists():
        parts.insert(0, pd.read_parquet(out_path))
    before = len(parts[-1])
    combined = pd.concat(parts, ignore_index=True).drop_duplicates(key_cols, keep="last")
    combined.to_parquet(out_path, index=False)
    partial.unlink()
    logger.info("Recovered %d rows from %s -> %s now has %d rows", before, partial, out_path, len(combined))


def stage_pm3(cand: pd.DataFrame, out_path: Path, args) -> None:
    key = ["inchikey", "charge_state", "charged_atom_idx"]
    _reconcile_partial(out_path, key)
    done = _done_keys(out_path, key)
    jobs = [row._asdict() for row in cand.itertuples(index=False)
            if (row.inchikey, row.charge_state, row.charged_atom_idx) not in done]
    logger.info("Stage PM3: %d candidates, %d already done, %d to run", len(cand), len(done), len(jobs))
    _run_pool(jobs, out_path, _PM3_SCHEMA, append=bool(done),
              fn=_pm3_one, fn_args=(args.orca_bin, args.pm3_timeout_seconds, args.scratch_dir),
              n_jobs=args.n_jobs, chunk=args.chunk_size)


# --------------------------------------------------------------------------- #
# Stage 2: DFT single point on selected molecules
# --------------------------------------------------------------------------- #
_DFT_SCHEMA = pa.schema([
    ("inchikey", pa.string()), ("charge_state", pa.int8()), ("charged_atom_idx", pa.int32()),
    ("dft_energy_hartree", pa.float64()), ("dft_method", pa.string()), ("dft_status", pa.string()),
])


def select_dft_molecules(pm3: pd.DataFrame, cand: pd.DataFrame, args) -> pd.DataFrame:
    """Return the pm3 rows for molecules that need DFT. Selection is per
    (inchikey, charge_state): near-tie at PM3, OR carries a clean secondary peak
    (gap), OR is an anion (PM3 unreliable). DFT is then run for ALL candidates of
    a selected molecule so energies stay homogeneous within a ranking."""
    ok = pm3[(pm3.pm3_status.isin(["ok", "pm3_not_converged"])) & pm3.pm3_energy_hartree.notna()].copy()
    gap_flag = (cand.groupby(["inchikey", "charge_state"])["has_clean_secondary_ccs"].any()
                if "has_clean_secondary_ccs" in cand.columns else None)
    selected = []
    for (ik, cs), g in ok.groupby(["inchikey", "charge_state"]):
        if len(g) < 2:
            continue  # single candidate -> nothing to rank, keep PM3
        e = np.sort(g.pm3_energy_hartree.to_numpy())
        top2_gap_kcal = (e[1] - e[0]) * HARTREE_KCAL
        is_anion = cs == -1
        is_gap = bool(gap_flag.get((ik, cs), False)) if gap_flag is not None else False
        near_tie = top2_gap_kcal < args.near_tie_kcal
        if near_tie or is_gap or (is_anion and args.dft_anions):
            selected.append(g)
    return pd.concat(selected, ignore_index=True) if selected else ok.iloc[0:0]


def _dft_one(job: dict, method_pos: str, method_neg: str, orca_bin: str, timeout_s: int,
             scratch_root: Optional[str]) -> dict:
    r = {k: job[k] for k in ("inchikey", "charge_state", "charged_atom_idx")}
    method = method_neg if job["charge_state"] == -1 else method_pos
    try:
        if not job.get("opt_xyz"):
            return {**r, "dft_energy_hartree": None, "dft_method": method, "dft_status": "no_geometry"}
        e, st = run_dft_sp(job["opt_xyz"], int(job["charge_state"]), method, orca_bin, timeout_s, scratch_root)
        return {**r, "dft_energy_hartree": e, "dft_method": method, "dft_status": st}
    except Exception as exc:
        return {**r, "dft_energy_hartree": None, "dft_method": method,
                "dft_status": f"worker_exception:{type(exc).__name__}"}


def stage_dft(pm3_path: Path, cand: pd.DataFrame, out_path: Path, args) -> None:
    pm3 = pd.read_parquet(pm3_path)
    sel = select_dft_molecules(pm3, cand, args)
    key = ["inchikey", "charge_state", "charged_atom_idx"]
    _reconcile_partial(out_path, key)
    done = _done_keys(out_path, key)
    jobs = [r._asdict() for r in sel.itertuples(index=False)
            if (r.inchikey, r.charge_state, r.charged_atom_idx) not in done]
    logger.info("Stage DFT: %d candidates in selected molecules, %d done, %d to run (pos=%s neg=%s)",
                len(sel), len(done), len(jobs), args.dft_method_pos, args.dft_method_neg)
    _run_pool(jobs, out_path, _DFT_SCHEMA, append=bool(done),
              fn=_dft_one, fn_args=(args.dft_method_pos, args.dft_method_neg,
                                    args.orca_bin, args.dft_timeout_seconds, args.scratch_dir),
              n_jobs=args.n_jobs, chunk=args.chunk_size)


# --------------------------------------------------------------------------- #
# Stage 3: rank + protomer/conformer signals
# --------------------------------------------------------------------------- #
def stage_rank(pm3_path: Path, dft_path: Optional[Path], cand: pd.DataFrame,
               out_cand: Path, out_mol: Path, args) -> None:
    pm3 = pd.read_parquet(pm3_path)
    dft = pd.read_parquet(dft_path) if (dft_path and Path(dft_path).exists()) else \
        pd.DataFrame(columns=["inchikey", "charge_state", "charged_atom_idx", "dft_energy_hartree", "dft_method", "dft_status"])
    df = pm3.merge(dft, on=["inchikey", "charge_state", "charged_atom_idx"], how="left")
    meta_cols = [c for c in ("parent_smiles", "site_names", "pka_mean", "n_candidates",
                             "has_clean_secondary_ccs") if c in cand.columns]
    df = df.merge(cand[["inchikey", "charge_state", "charged_atom_idx", *meta_cols]].drop_duplicates(
        ["inchikey", "charge_state", "charged_atom_idx"]),
        on=["inchikey", "charge_state", "charged_atom_idx"], how="left")
    df["site_element"] = df["site_names"].map(_site_element) if "site_names" in df else "?"
    df["adduct"] = df["charge_state"].map({1: "[M+H]+", -1: "[M-H]-"})

    out_rows, mol_rows = [], []
    for (ik, cs), g in df.groupby(["inchikey", "charge_state"]):
        g = g.copy()
        # homogeneous final energy: all-DFT if every candidate has a DFT energy, else all-PM3
        all_dft = g["dft_energy_hartree"].notna().all() and len(g) > 0
        g["energy_level"] = "dft" if all_dft else "pm3"
        g["final_energy_hartree"] = g["dft_energy_hartree"] if all_dft else g["pm3_energy_hartree"]
        valid = g[g["final_energy_hartree"].notna()].copy()
        if valid.empty:
            g["rel_energy_kcal"] = np.nan; g["boltzmann_pop"] = np.nan; g["abundance_rank"] = -1
            out_rows.append(g); continue
        emin = valid["final_energy_hartree"].min()
        g["rel_energy_kcal"] = (g["final_energy_hartree"] - emin) * HARTREE_KCAL
        w = np.exp(-g["rel_energy_kcal"].clip(lower=0) / (args.temperature / 298.15 * KT_298_KCAL))
        w = w.where(g["final_energy_hartree"].notna(), 0.0)
        g["boltzmann_pop"] = w / w.sum() if w.sum() > 0 else np.nan
        g["abundance_rank"] = g["rel_energy_kcal"].rank(method="min").astype("Int64")
        out_rows.append(g)

        vv = g[g["final_energy_hartree"].notna()].sort_values("rel_energy_kcal")
        pops = vv["boltzmann_pop"].to_numpy()
        n_pop = int((pops > args.pop_threshold).sum())
        top2_gap = float(vv["rel_energy_kcal"].iloc[1]) if len(vv) >= 2 else np.nan
        cross_elem = bool(len(vv) >= 2 and vv["site_element"].iloc[0] != vv["site_element"].iloc[1])
        mol_rows.append({
            "inchikey": ik, "charge_state": cs, "adduct": "[M+H]+" if cs == 1 else "[M-H]-",
            "n_candidates_ranked": len(vv), "energy_level": g["energy_level"].iloc[0],
            "most_abundant_charged_atom_idx": int(vv["charged_atom_idx"].iloc[0]) if len(vv) else -1,
            "most_abundant_site_element": vv["site_element"].iloc[0] if len(vv) else "?",
            "n_populated": n_pop, "top2_energy_gap_kcal": top2_gap,
            "cross_element_competition": cross_elem,
            "has_clean_secondary_ccs": bool(g["has_clean_secondary_ccs"].iloc[0]) if "has_clean_secondary_ccs" in g else False,
            # protomer-plausible = >=2 protomers coexist; a has_gap molecule that is
            # NOT protomer-plausible has a secondary peak better explained as a conformer
            "protomer_plausible": n_pop >= 2,
        })

    cand_out = pd.concat(out_rows, ignore_index=True)
    cand_out.to_parquet(out_cand, index=False)
    mol_out = pd.DataFrame(mol_rows)
    mol_out.to_parquet(out_mol, index=False)
    logger.info("Stage rank: wrote %d candidate rows -> %s ; %d molecule rows -> %s",
                len(cand_out), out_cand, len(mol_out), out_mol)
    if len(mol_out):
        gapm = mol_out[mol_out["has_clean_secondary_ccs"]]
        if len(gapm):
            logger.info("Gap molecules: %d; protomer-plausible %.0f%%, conformer-suspect %.0f%%",
                        len(gapm), 100 * gapm["protomer_plausible"].mean(),
                        100 * (~gapm["protomer_plausible"]).mean())


# --------------------------------------------------------------------------- #
# shared parallel pool + chunked writer
# --------------------------------------------------------------------------- #
def _run_pool(jobs, out_path, schema, append, fn, fn_args, n_jobs, chunk):
    """Run `fn(job, *fn_args)` over jobs in a process pool, writing results to a
    parquet in chunks. `fn` must be a top-level (picklable) function; futures are
    submitted in bounded waves so memory stays flat at ~800k jobs."""
    if not jobs:
        logger.info("Nothing to do for %s", out_path)
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".partial.parquet") if append else out_path
    writer = pq.ParquetWriter(tmp, schema)
    buf = []
    block = max(chunk * 10, n_jobs * 50)

    def flush():
        nonlocal buf
        if buf:
            writer.write_table(pa.Table.from_pandas(pd.DataFrame(buf), schema=schema, preserve_index=False))
            buf = []
    try:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            done_n = 0
            for start in range(0, len(jobs), block):
                futs = [ex.submit(fn, j, *fn_args) for j in jobs[start:start + block]]
                for fut in as_completed(futs):
                    buf.append(fut.result())
                    done_n += 1
                    if len(buf) >= chunk:
                        flush()
                    if done_n % 25 == 0:
                        logger.info("  %d / %d done", done_n, len(jobs))
                flush()
    finally:
        writer.close()
    if append:  # concatenate the new partial onto the existing output
        combined = pd.concat([pd.read_parquet(out_path), pd.read_parquet(tmp)], ignore_index=True)
        combined.to_parquet(out_path, index=False)
        tmp.unlink()


def load_or_build_candidates(data_path: Path, candidates_path: Path,
                             limit: Optional[int], chunk_size: int,
                             force: bool) -> pd.DataFrame:
    """Return the protonation-candidate table, enumerating from the ORIGINAL
    Enveda data if the cache is absent.

    Enumeration is delegated to enumerate_protonation_candidates.py (the same
    chain that built enveda_widened.parquet), never reimplemented here. The
    heavy import (dimorphite-dl et al.) happens only when the cache must be
    built, so resumed stages that already have the cache stay lightweight.
    """
    if candidates_path.exists() and not force:
        logger.info("Reusing existing candidate cache %s (delete or --force-enumerate to rebuild)",
                    candidates_path)
        return pd.read_parquet(candidates_path)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Candidate cache {candidates_path} not found and raw data {data_path} "
            "is missing; pass --data (original Enveda file) or --candidates (pre-built parquet)."
        )

    logger.info("Enumerating protonation candidates from raw data %s -> %s", data_path, candidates_path)
    import enumerate_protonation_candidates as epc  # heavy; import only when building

    spectra_df = epc.load_spectra(data_path, limit_molecules=limit)
    multi = epc.build_secondary_species_table(spectra_df)
    mol_df = epc.build_molecule_table(spectra_df, multi)
    candidates_path.parent.mkdir(parents=True, exist_ok=True)
    stats = epc.run_candidate_generation(mol_df, candidates_path, chunk_size)
    logger.info("Enumeration done: %d molecules processed", stats.get("molecules_processed", 0))
    return pd.read_parquet(candidates_path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/Enveda-180/enveda-180.jsonl.gz"),
                    help="ORIGINAL Enveda spectra file (.jsonl.gz or -dedup.csv); candidates are "
                         "enumerated from this via enumerate_protonation_candidates.py when the "
                         "--candidates cache is absent")
    ap.add_argument("--candidates", type=Path, default=Path("outputs/protomer_energies/candidates.parquet"),
                    help="Candidate-table cache; reused if present, otherwise built from --data. "
                         "Point at outputs/protonation_candidates/enveda_widened.parquet to reuse "
                         "the pre-built set and skip enumeration")
    ap.add_argument("--force-enumerate", action="store_true",
                    help="Rebuild the candidate cache from --data even if --candidates exists")
    ap.add_argument("--stage", choices=["pm3", "dft", "rank", "all"], default="all")
    ap.add_argument("--pm3-out", type=Path, default=Path("outputs/protomer_energies/pm3.parquet"))
    ap.add_argument("--dft-out", type=Path, default=Path("outputs/protomer_energies/dft.parquet"))
    ap.add_argument("--rank-out", type=Path, default=Path("outputs/protomer_energies/ranked_candidates.parquet"))
    ap.add_argument("--mol-out", type=Path, default=Path("outputs/protomer_energies/ranked_molecules.parquet"))
    ap.add_argument("--orca-bin", default=DEFAULT_ORCA_BIN)
    ap.add_argument("--dft-method-pos", default="r2SCAN-3c",
                    help="ORCA keyword for cation single points")
    ap.add_argument("--dft-method-neg", default="r2SCAN def2-TZVPD D4 RIJCOSX def2/J",
                    help="ORCA keyword for anion single points (needs DIFFUSE functions; r2SCAN-3c has none)")
    ap.add_argument("--near-tie-kcal", type=float, default=3.0,
                    help="PM3 top-2 gap below which a molecule is sent to DFT")
    ap.add_argument("--dft-anions", action="store_true", default=True,
                    help="Always DFT multi-candidate anions (PM3 anion energies unreliable)")
    ap.add_argument("--no-dft-anions", dest="dft_anions", action="store_false")
    ap.add_argument("--temperature", type=float, default=298.15)
    ap.add_argument("--pop-threshold", type=float, default=0.05)
    ap.add_argument("--pm3-timeout-seconds", type=int, default=300)
    ap.add_argument("--dft-timeout-seconds", type=int, default=1200)
    ap.add_argument("--n-jobs", type=int, default=min(16, (os.cpu_count() or 2)))
    ap.add_argument("--chunk-size", type=int, default=200)
    ap.add_argument("--scratch-dir", default="/dev/shm" if Path("/dev/shm").exists() else None)
    ap.add_argument("--limit", type=int, default=None, help="cap candidates (smoke testing)")
    args = ap.parse_args(argv)

    cand = load_or_build_candidates(args.data, args.candidates, args.limit,
                                    args.chunk_size, args.force_enumerate)
    cand = cand[cand["candidate_smiles"].notna() & cand["charged_atom_idx"].notna()].copy()
    cand["charged_atom_idx"] = cand["charged_atom_idx"].astype("int32")
    cand["charge_state"] = cand["charge_state"].astype("int8")
    if args.limit:
        keep = cand.drop_duplicates("inchikey").head(args.limit)["inchikey"]
        cand = cand[cand["inchikey"].isin(keep)]
    logger.info("Loaded %d candidates (%d molecules)", len(cand), cand["inchikey"].nunique())

    if args.stage in ("pm3", "all"):
        stage_pm3(cand, args.pm3_out, args)
    if args.stage in ("dft", "all"):
        stage_dft(args.pm3_out, cand, args.dft_out, args)
    if args.stage in ("rank", "all"):
        stage_rank(args.pm3_out, args.dft_out, cand, args.rank_out, args.mol_out, args)


if __name__ == "__main__":
    main()
