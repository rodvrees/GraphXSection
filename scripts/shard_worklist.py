#!/usr/bin/env python3
"""
Split a QM work-list Parquet into N shards for parallel/crash-resilient runs
of predict_candidate_ccs_dft.py or predict_candidate_ccs_md.py. Shards are
split by INCHIKEY (a molecule's candidates always stay together in one shard
-- required, since matching/anchoring is per-molecule), not by row, and
written as out_00.parquet, out_01.parquet, ... matching the naming already
used for the DFT-static run.

Molecules are shuffled (seeded) before splitting so shard sizes are balanced
even though candidates-per-molecule varies; a contiguous (unshuffled) split
would let one shard land mostly single-candidate molecules and another mostly
5-6-candidate ones.

Why shard at all: neither prediction script supports --resume, and the output
Parquet only gets a valid footer on clean close -- a crash loses the whole
run. Splitting into shards bounds the loss to one shard's candidates and lets
you run shards in parallel (one process per shard) instead of one process
with --n-jobs workers sharing a single all-or-nothing output file.

Usage:
    python scripts/shard_worklist.py \\
        --input outputs/protonation_candidates/qm_worklist_band_9pct.parquet \\
        --output-dir outputs/protonation_candidates/shards_9pct \\
        --num-shards 10
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, required=True, help="Work-list Parquet to shard (e.g. qm_worklist_band_9pct.parquet)")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--num-shards", type=int, required=True)
    p.add_argument("--seed", type=int, default=0, help="Shuffle seed for balanced shard sizes (default 0)")
    args = p.parse_args()

    df = pd.read_parquet(args.input)
    inchikeys = df["inchikey"].unique()
    rng = np.random.default_rng(args.seed)
    rng.shuffle(inchikeys)
    shard_of = {ik: i % args.num_shards for i, ik in enumerate(inchikeys)}
    df["_shard"] = df["inchikey"].map(shard_of)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for shard_idx in range(args.num_shards):
        shard_df = df[df["_shard"] == shard_idx].drop(columns=["_shard"])
        out_path = args.output_dir / f"out_{shard_idx:02d}.parquet"
        shard_df.to_parquet(out_path, index=False)
        logger.info(
            "Shard %02d: %d molecules / %d candidates -> %s",
            shard_idx, shard_df["inchikey"].nunique(), len(shard_df), out_path,
        )

    logger.info(
        "Done: %d molecules / %d candidates across %d shards (%s)",
        df["inchikey"].nunique(), len(df), args.num_shards, args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
