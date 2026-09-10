#!/usr/bin/env bash
# Run the column MLP acceptance runner on every eval-mode fixture and collect
# reports under benchmarks/results/.  Exit status is nonzero if any fixture is
# not accepted.  Usage: benchmarks/run_all.sh [fixture_dir]
set -u
DIR="${1:-data/fixtures}"
OUT="benchmarks/results"
mkdir -p "$OUT"
status=0
for f in "$DIR"/column_mlp_*_float64.h5 "$DIR"/column_mlp_*_float32.h5; do
  [ -e "$f" ] || continue
  echo "=== $f"
  /usr/bin/time -p julia --project=. cases/column_mlp/validate.jl "$f" "$OUT" 2>&1 | grep -v "absl\|cpp_gen\|^\s*\[\|^\s*@\|Warning\|└" | tee "$OUT/$(basename "$f" .h5).log" | tail -6
  rc=${PIPESTATUS[0]}
  [ "$rc" -eq 0 ] || status=1
done
exit $status
