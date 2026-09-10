#!/usr/bin/env bash
# Run the column MLP acceptance runner on every eval-mode fixture and collect
# reports under benchmarks/results/.  Exit status is nonzero if any fixture is
# not accepted.  Usage: benchmarks/run_all.sh [fixture_dir]
set -uo pipefail
shopt -s nullglob
DIR="${1:-data/fixtures}"
OUT="benchmarks/results"
fixtures=("$DIR"/column_mlp_*_float64.h5 "$DIR"/column_mlp_*_float32.h5)
if [ "${#fixtures[@]}" -eq 0 ]; then
  printf 'No eval fixtures found in %s; no acceptance checks ran.\n' "$DIR" >&2
  exit 1
fi
mkdir -p "$OUT" || exit 1
status=0
rejected=0
for f in "${fixtures[@]}"; do
  printf '=== %s\n' "$f"
  # Keep the complete diagnostic log; only truncate the terminal summary.
  # pipefail propagates both Julia failures and failed log writes.
  if /usr/bin/time -p julia --project=. cases/column_mlp/validate.jl "$f" "$OUT" 2>&1 |
      tee "$OUT/$(basename "$f" .h5).log" | tail -6; then
    :
  else
    status=1
    rejected=$((rejected + 1))
  fi
done
printf 'Ran %d eval fixtures; %d not accepted or failed to run.\n' "${#fixtures[@]}" "$rejected"
exit $status
