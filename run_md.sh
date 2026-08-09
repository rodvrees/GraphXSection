cd /home/robbe/GraphXSection
mkdir -p outputs/protonation_candidates/md_results

for i in 00 01 02 03 04 05 06 07 08 09; do
  ~/.pyenv/versions/GXS/bin/python scripts/predict_candidate_ccs_md.py \
    --input  outputs/protonation_candidates/shards_9pct_md/out_${i}.parquet \
    --output outputs/protonation_candidates/md_results/out_${i}_result.parquet \
    --n-molecules 3000 \
    --n-jobs 80 \
    --mobcal-concurrency 20 \
    --mobcal-threads 8 \
    --mobcal-core-block-size 8 \
    --mobcal-timeout-seconds 600 \
    > outputs/protonation_candidates/md_results/out_${i}.log 2>&1
done