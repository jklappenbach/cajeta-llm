#!/usr/bin/env bash
# census-fallout.sh LOG
#
# From one run-tests.sh log, the census rows that need a hand: the SKIP entries
# that went STALE (a tracked kernel ran, prune its entry), the kernels that RAN
# with no value check (bracket a reference comparison with ValueCheck, or name
# the item in KernelCensus.unchecked), and anything UNCOVERED. Reads the
# `census-row` lines KernelCensus prints (backend, kernel, class, launches,
# checks, manifest, note).
log="$1"
rows() { grep "^census-row"$'\t' "$log" | cut -f2-; }
echo "== $(grep -o 'kernel census ([a-z]*): .*' "$log" | tail -1)"
for cls in STALE-SKIP STALE-UNCHECKED UNCOVERED RAN-UNCHECKED; do
    n=$(rows | awk -F'\t' -v c="$cls" '$3==c' | wc -l)
    echo "-- $cls ($n):"
    rows | awk -F'\t' -v c="$cls" '$3==c { printf "   %s  launches=%s  %s\n", $2, $4, $7 }'
done
echo "-- CHECKED: $(rows | awk -F'\t' '$3=="CHECKED"' | wc -l)   SKIP: $(rows | awk -F'\t' '$3=="SKIP"' | wc -l)"
