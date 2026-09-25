#!/bin/bash
# Dry-run every protocol x model x task. Prints only failures.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
fail=0; n=0
run() { n=$((n+1)); DRY_RUN=1 "$@" >/dev/null 2>&1 || { echo "FAIL: $*"; fail=$((fail+1)); }; }

for m in dino jepa lejepa ntp mae diffusion softclt; do
  for t in forecast anomaly classify; do
    run ./scripts/run_per_model.sh     "$m" "$t" 123
    run ./scripts/run_equal_budget.sh  "$m" "$t" 123
    run ./scripts/run_long_pretrain.sh "$m" "$t" 123
    for c in synthetic synthetic_small hybrid; do
      run ./scripts/run_corpus.sh "$m" "$c" "$t" 123
    done
    for h in linear mlp finetune; do
      run ./scripts/run_probe_head.sh "$m" "$h" "$t" 123
    done
  done
  run ./scripts/run_indomain.sh "$m" 123
done
for t in forecast anomaly classify; do
  run ./scripts/run_random_baseline.sh "$t" 123
done
for f in isotropy drift tsne gaussianity all; do
  run ./scripts/run_figures.sh "$f"
done
for s in full mix subset all; do
  run ./scripts/run_synthetic_generation.sh "$s"
done
run ./scripts/launch.sh "0 1" ./scripts/run_per_model.sh %M forecast %S

echo "$n dry-runs, $fail failures"
