#!/usr/bin/env python3
"""
Filter the full Step 1/2 enumeration output (qm_worklist.parquet) down to
molecules with at least one clean, dimer-filtered secondary CCS peak whose
gap from the primary falls in [--band-low, --band-high]. This is the QM
work-list handed to predict_candidate_ccs_dft.py / predict_candidate_ccs_md.py.

clean_secondary_peaks_worklist.parquet already excludes dimers (ratio <=
DIMER_THRESH); this script only applies the gap band on top of that.

Usage:
    python scripts/build_qm_worklist_band.py \\
        --candidates outputs/protonation_candidates/qm_worklist.parquet \\
        --peaks outputs/protonation_candidates/clean_secondary_peaks_worklist.parquet \\
        --band-low 0.02 --band-high 0.09 \\
        --output outputs/protonation_candidates/qm_worklist_band_9pct.parquet
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--candidates", type=Path, required=True, help="Full enumeration output (qm_worklist.parquet)")
    p.add_argument("--peaks", type=Path, required=True, help="clean_secondary_peaks_worklist.parquet (dimer-filtered)")
    p.add_argument("--band-low", type=float, default=0.02, help="Minimum |ratio-1| gap to keep (default 0.02)")
    p.add_argument("--band-high", type=float, default=0.15, help="Maximum |ratio-1| gap to keep (default 0.15)")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    cand = pd.read_parquet(args.candidates)
    peaks = pd.read_parquet(args.peaks)
    peaks["gap_frac"] = (peaks["ratio"] - 1.0).abs()

    in_band = peaks[(peaks["gap_frac"] >= args.band_low) & (peaks["gap_frac"] <= args.band_high)]
    keep_inchikeys = set(in_band["inchikey"].unique())

    out = cand[cand["inchikey"].isin(keep_inchikeys)].copy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)

    logger.info(
        "Band [%.0f%%, %.0f%%]: %d molecules / %d candidates (from %d molecules / %d candidates unfiltered)",
        args.band_low * 100, args.band_high * 100,
        out["inchikey"].nunique(), len(out),
        cand["inchikey"].nunique(), len(cand),
    )
    logger.info("Wrote %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
