#!/usr/bin/env python3
"""
Step 2 of the (revised) protomer plan: a cheap, a-priori screen that decides
which multi-peak molecules are worth expensive QM validation (Step 3), so
compute is not wasted on multi-peak molecules that cannot be protomers.

Consumes the WIDENED enumeration from enumerate_protonation_candidates.py
(N/O/S sites) and keeps a (inchikey, charge_state) group only if ALL of:

  1. it is multi-peak (`has_clean_secondary_ccs` -- a secondary CCS peak in the
     protomer/conformer band, not a dimer), i.e. an actual Model 2 target;
  2. it has >= --min-sites distinct enumerated protonation sites (after
     symmetry collapse; `n_candidates`). A molecule with one site cannot form a
     protomer -- the extra peak must be a conformer/dimer/artifact;
  3. it has >= --min-relevant "protomer-relevant" sites, where relevant =
     basic nitrogen (amine_N / aromatic_N / dimorphite N classes) OR a carbonyl
     oxygen (carbonyl_O). This is the pattern behind real protomers (a basic N
     plus a conjugation-stabilizable carbonyl-O, or two basic N's) and drops
     molecules whose only extra sites are weakly-basic ether/hydroxyl/furan
     O or thioether S.

Deliberately NOT used as screen criteria (per the danofloxacin ground-truth
finding): gas-phase energy / proton-affinity gaps (kinetic trapping from
solution means a large gap does not rule out a second observed protomer), and a
solution-pKa anchor threshold (benzocaine, a classic protomer, has max pKa ~3.9;
and every input molecule is already observed as [M+H]+ so the "is it basic
enough" question is moot). Thresholds are CLI-tunable and were calibrated on a
known panel -- KEEP: danofloxacin, ciprofloxacin, benzocaine, 4-aminobenzoic
acid; DROP: phenethylamine, triethylamine (single-site), glucose (all ether-O).

Output: the surviving candidate rows (the QM work-list for
predict_candidate_ccs_dft.py), same schema as the input parquet.

Usage:
    python scripts/screen_protomer_candidates.py \\
        --input outputs/protonation_candidates/enveda_dedup_widened.parquet \\
        --output outputs/protonation_candidates/qm_worklist.parquet
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# A site is "protomer-relevant" if its site_names set contains any of these.
# Every nitrogen site carries amine_N or aromatic_N (added by
# enumerate_protonation_candidates.site_type_name); carbonyl_O marks the
# conjugation-stabilizable oxygen protomers (e.g. danofloxacin's keto-O+).
DEFAULT_RELEVANT_TOKENS = ("amine_N", "aromatic_N", "carbonyl_O")


def _has_relevant(site_names: object, tokens: tuple[str, ...]) -> bool:
    if not isinstance(site_names, str):
        return False
    return any(tok in site_names for tok in tokens)


def _is_basic_n(sn: object) -> bool:
    return isinstance(sn, str) and ("amine_N" in sn or "aromatic_N" in sn)


def _is_carbonyl_o(sn: object) -> bool:
    return isinstance(sn, str) and "carbonyl_O" in sn


def screen(
    df: pd.DataFrame,
    charge_state: int,
    min_sites: int,
    min_relevant: int,
    relevant_tokens: tuple[str, ...],
    require_clean_secondary: bool,
    require_both: bool = True,
    relevant_only: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Select the QM work-list. Panel-safe default (require_both=True): a group
    survives only if it has >=1 basic-N AND >=1 carbonyl-O candidate -- the
    N-vs-O protomer competition seen in every confirmed case (danofloxacin,
    benzocaine, ...). relevant_only=True then emits, per surviving group, only
    the protomer-relevant candidates (basic-N or carbonyl-O), dropping the
    weakly-basic ether/hydroxyl/aromatic-O and thioether-S sites that inflate
    QM cost without being plausible protomers ("capping" by site type). This
    keeps #candidates >= #peaks for ~all groups (coverage) while cutting the
    per-group candidate count. require_both=False falls back to the looser
    >=min_relevant criterion.
    """
    pol = df[(df["charge_state"] == charge_state) & df["candidate_smiles"].notna()].copy()
    pol["_relevant"] = pol["site_names"].apply(lambda s: _has_relevant(s, relevant_tokens))
    pol["_basic_n"] = pol["site_names"].apply(_is_basic_n)
    pol["_carbonyl_o"] = pol["site_names"].apply(_is_carbonyl_o)

    per_group = pol.groupby(["inchikey", "charge_state"]).agg(
        n_sites=("candidate_smiles", "size"),
        n_relevant=("_relevant", "sum"),
        n_basic_n=("_basic_n", "sum"),
        n_carbonyl_o=("_carbonyl_o", "sum"),
        multi_peak=("has_clean_secondary_ccs", "max"),
    )

    keep = per_group["n_sites"] >= min_sites
    if require_both:
        keep &= (per_group["n_basic_n"] >= 1) & (per_group["n_carbonyl_o"] >= 1)
    else:
        keep &= per_group["n_relevant"] >= min_relevant
    if require_clean_secondary:
        keep &= per_group["multi_peak"].fillna(False).astype(bool)

    survivor_keys = per_group[keep].index
    survivors = pol.set_index(["inchikey", "charge_state"]).loc[survivor_keys].reset_index()
    if relevant_only:
        survivors = survivors[survivors["_relevant"]]
    survivors = survivors.drop(columns=["_relevant", "_basic_n", "_carbonyl_o"])

    stats = {
        "input_groups": int(per_group.shape[0]),
        "input_candidates": int(len(pol)),
        "multi_peak_groups": int(per_group["multi_peak"].fillna(False).astype(bool).sum()),
        "survivor_groups": int(len(survivor_keys)),
        "survivor_candidates": int(len(survivors)),
    }
    return survivors, stats


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="Widened enumeration parquet (enumerate_protonation_candidates.py)")
    parser.add_argument("--output", type=Path, required=True, help="Output QM work-list parquet (survivor candidate rows)")
    parser.add_argument("--charge-state", type=int, default=1, help="Polarity to screen (default 1 = [M+H]+ protomers)")
    parser.add_argument("--min-sites", type=int, default=2, help="Minimum distinct enumerated sites (default 2)")
    parser.add_argument("--min-relevant", type=int, default=2, help="Minimum protomer-relevant sites (default 2)")
    parser.add_argument(
        "--relevant-tokens", type=str, default=",".join(DEFAULT_RELEVANT_TOKENS),
        help="Comma-separated site_name substrings counted as protomer-relevant",
    )
    parser.add_argument(
        "--no-require-secondary", action="store_true",
        help="Do not require a clean secondary CCS peak (screen structure only; for calibration/testing)",
    )
    parser.add_argument(
        "--no-require-both", action="store_true",
        help="Do not require both a basic-N and a carbonyl-O site; fall back to the looser >=min-relevant rule",
    )
    parser.add_argument(
        "--keep-all-sites", action="store_true",
        help="Emit all N/O/S candidate rows for surviving groups (default: only protomer-relevant "
             "basic-N / carbonyl-O candidates, dropping weakly-basic ether/hydroxyl/aromatic-O and S)",
    )
    args = parser.parse_args(argv)

    tokens = tuple(t.strip() for t in args.relevant_tokens.split(",") if t.strip())
    df = pd.read_parquet(args.input)
    survivors, stats = screen(
        df, args.charge_state, args.min_sites, args.min_relevant, tokens,
        require_clean_secondary=not args.no_require_secondary,
        require_both=not args.no_require_both,
        relevant_only=not args.keep_all_sites,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    survivors.to_parquet(args.output, index=False)

    logger.info("Screen summary (charge_state=%+d):", args.charge_state)
    for k, v in stats.items():
        logger.info("  %s: %d", k, v)
    if stats["multi_peak_groups"]:
        logger.info(
            "  survivors are %.1f%% of multi-peak groups",
            100 * stats["survivor_groups"] / stats["multi_peak_groups"],
        )
    logger.info("Wrote QM work-list to %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
