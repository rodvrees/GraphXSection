mkdir -p logs outputs/protonation_candidates/band_shards

# 1. split into 10 per-candidate shards (safe: Step 4 regroups by molecule after concat)
python - <<'PY'
import pandas as pd, numpy as np
df = pd.read_parquet("outputs/protonation_candidates/qm_worklist_band.parquet")
for i, part in enumerate(np.array_split(df, 10)):
    part.to_parquet(f"outputs/protonation_candidates/band_shards/in_{i:02d}.parquet", index=False)
print("wrote 10 shards,", len(df), "candidates")
PY

# 2. run shards sequentially. Each mobcal_shm call (--mobcal-threads 1) pins 2
#    cores, and every worker must get a DISJOINT core block or the calls collide
#    and stall to a timeout, so --n-jobs * --mobcal-core-block-size must stay <=
#    the core count (256 here): 128 * 2 = 256, an exact collision-free fit.
nohup bash -c '
for s in outputs/protonation_candidates/band_shards/in_*.parquet; do
  out="outputs/protonation_candidates/band_shards/out_$(basename "$s" | sed "s/in_//")"
  [ -f "$out" ] && { echo "skip done $out"; continue; }
  echo "=== $s -> $out ==="
  if ! python scripts/predict_candidate_ccs_dft.py \
      --candidates "$s" --output "$out" \
      --dft-method r2SCAN-3c --n-jobs 128 --mobcal-threads 1 --mobcal-core-block-size 2 \
      --max-consecutive-failures 50; then
    echo "ABORT: $s hit the 50-consecutive-failure limit (or errored); removing partial $out and stopping the whole loop."
    rm -f "$out"
    break
  fi
done
' > logs/qm_band.log 2>&1 &
echo "PID $!  |  tail -f logs/qm_band.log"