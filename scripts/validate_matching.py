#!/usr/bin/env python3
"""
Compares build_protomer_labels.py's positional candidate<->peak matching
(rank-1 candidate by proton-affinity proxy paired with rank-1 peak by
intensity, and so on -- see that script's docstring) against a nearest-CCS
bipartite match using predict_candidate_ccs_md.py's independent, ORCA-MD +
MOBCAL-derived predicted CCS per candidate.

For every (inchikey, charge_state) group present in both inputs:
  - "positional": model2.parquet's existing candidate_smiles <-> observed_ccs
    pairing (one row per candidate already).
  - "nearest_ccs": candidates re-paired to the SAME observed_ccs values by
    solving the assignment problem (scipy.optimize.linear_sum_assignment) on
    the |predicted_ccs - observed_ccs| cost matrix.

Reports, per group, whether the two matchings agree (same permutation) or
not, plus the assignment cost of each so a molecule where nearest-CCS
matching is a much better fit than positional matching (or vice versa) is
easy to spot. Only groups with >=2 candidates present in the MD-predicted
CCS input are considered -- single-candidate groups have nothing to
re-match.

Usage (either environment with pandas/scipy/pyarrow works, e.g. the "Waters"
pyenv env or GXS_torch):
    python scripts/validate_matching.py \\
        --model2-input outputs/protonation_candidates/model2.parquet \\
        --predicted-ccs-input outputs/protonation_candidates/candidate_ccs_md.parquet \\
        --output outputs/protonation_candidates/matching_validation.parquet
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def _rank_column(df: pd.DataFrame) -> str:
    """build_protomer_labels.py's rank column has gone by more than one name
    across runs of this pipeline (confirmed directly: a model2.parquet seen
    in practice used "pka_rank" + "weak_pka_rank"/"unambiguous_single_site"
    match_confidence values, rather than this repo's current
    build_protomer_labels.py, which still writes "candidate_rank" +
    "ranking_source"/"weak_rank_<source>"). Accept either rather than
    hard-failing on a KeyError."""
    for candidate in ("candidate_rank", "pka_rank"):
        if candidate in df.columns:
            return candidate
    raise KeyError(
        f"No recognized rank column (tried candidate_rank, pka_rank) in columns: {df.columns.tolist()}"
    )


def compare_group(group_key: tuple[str, int], model2_group: pd.DataFrame, predicted_group: pd.DataFrame) -> Optional[dict]:
    inchikey, charge_state = group_key

    merged = model2_group.merge(
        predicted_group[["candidate_smiles", "predicted_ccs", "predicted_ccs_std", "n_frames_used"]],
        on="candidate_smiles", how="inner",
    )
    if len(merged) < 2 or merged["predicted_ccs"].isna().any():
        return None

    rank_col = _rank_column(merged)
    merged = merged.sort_values(rank_col).reset_index(drop=True)
    n = len(merged)
    observed = merged["observed_ccs"].to_numpy()
    predicted = merged["predicted_ccs"].to_numpy()

    # Positional (existing heuristic): candidate i already paired with
    # observed[i] in model2.parquet (candidate_rank <-> peak intensity rank).
    positional_cost = float(np.sum(np.abs(predicted - observed)))

    # Nearest-CCS bipartite match: re-pair the SAME observed values to
    # candidates by minimizing total |predicted - observed|.
    cost_matrix = np.abs(predicted[:, None] - observed[None, :])
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    nearest_cost = float(cost_matrix[row_ind, col_ind].sum())
    nearest_assignment = col_ind[np.argsort(row_ind)]

    agrees = bool(np.array_equal(nearest_assignment, np.arange(n)))

    return {
        "inchikey": inchikey,
        "charge_state": charge_state,
        "n_candidates": n,
        "match_confidence": merged["match_confidence"].iloc[0],
        "ranking_source": merged["ranking_source"].iloc[0] if "ranking_source" in merged.columns else None,
        "positional_assignment_agrees_with_nearest_ccs": agrees,
        "positional_total_abs_error": positional_cost,
        "nearest_ccs_total_abs_error": nearest_cost,
        "mean_n_frames_used": float(merged["n_frames_used"].mean()),
        "observed_ccs": observed.tolist(),
        "predicted_ccs": predicted.tolist(),
        "candidate_smiles": merged["candidate_smiles"].tolist(),
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model2-input", type=Path, required=True)
    parser.add_argument("--predicted-ccs-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    model2_df = pd.read_parquet(args.model2_input)
    predicted_df = pd.read_parquet(args.predicted_ccs_input)
    predicted_df = predicted_df[predicted_df["status"] == "ok"]

    keys = predicted_df[["inchikey", "charge_state"]].drop_duplicates()
    logger.info("Comparing matching for %d molecule groups with MD-predicted CCS", len(keys))

    model2_by_group = dict(tuple(model2_df.groupby(["inchikey", "charge_state"])))
    predicted_by_group = dict(tuple(predicted_df.groupby(["inchikey", "charge_state"])))

    rows = []
    for key in keys.itertuples(index=False):
        group_key = (key.inchikey, key.charge_state)
        model2_group = model2_by_group.get(group_key)
        predicted_group = predicted_by_group.get(group_key)
        if model2_group is None or predicted_group is None:
            continue
        result = compare_group(group_key, model2_group, predicted_group)
        if result is not None:
            rows.append(result)

    result_df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_parquet(args.output, index=False)

    if len(result_df):
        agree_rate = result_df["positional_assignment_agrees_with_nearest_ccs"].mean()
        logger.info(
            "%d groups compared -- positional heuristic agrees with nearest-CCS matching in %.1f%% of cases",
            len(result_df), 100 * agree_rate,
        )
        logger.info(
            "Mean total |error|: positional=%.2f, nearest-CCS=%.2f",
            result_df["positional_total_abs_error"].mean(),
            result_df["nearest_ccs_total_abs_error"].mean(),
        )
    else:
        logger.warning("No comparable groups found -- check that the two inputs share (inchikey, charge_state) keys")

    logger.info("Wrote matching validation to %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
