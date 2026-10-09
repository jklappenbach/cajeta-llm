#!/usr/bin/env bash
# workgroup-reduce 6.1.2: build both arms and time them interleaved A B B A.
# Usage: run.sh <hand-tree> <moved-tree> <out-dir>
# Env: CAJETA (the v0.38.0 compiler), BE (amdgpu|nvptx), CP_DEPS, ROUNDS (default 4), SKIP_BUILD=1.
set -euo pipefail
hand=$(cd "$1" && pwd)
moved=$(cd "$2" && pwd)
out=$(mkdir -p "$3" && cd "$3" && pwd)
C=${CAJETA:?set CAJETA to the v0.38.0 compiler}
BE=${BE:?set BE to amdgpu or nvptx}
olla=$HOME/.olla
cp_deps=${CP_DEPS:-$olla/dev.cajeta.codec/0.8.6/dev.cajeta.codec-0.8.6.cja,$olla/dev.cajeta.jinja/0.1.5/dev.cajeta.jinja-0.1.5.cja,$olla/dev.cajeta.logging/0.8.5/dev.cajeta.logging-0.8.5.cja}
probe=$(cd "$(dirname "$0")" && pwd)
export TMPDIR=$out
export CAJETA_XPU_KERNEL_GATE=${GATE:-warn}
"$C" --version > "$out/compiler.txt"
if [ "${SKIP_BUILD:-0}" != 1 ]; then
  for arm in hand moved; do
    tree=$hand; [ "$arm" = moved ] && tree=$moved
    d=$out/$arm; b=$d/build; mkdir -p "$d/src/dev/cajeta/wgrtime" "$b"
    cp "$probe/WgrTiming.cajeta" "$d/src/dev/cajeta/wgrtime/"
    git -C "$tree" rev-parse HEAD > "$d/commit.txt"
    git -C "$tree" diff --stat >> "$d/commit.txt"
    (cd "$tree" && "$C" --emit=cja -o "$d/llama.cja" --classpath="$cp_deps" \
      dev.cajeta.llm.Llm.run src/main/cajeta "$b" >/dev/null 2>"$d/lib.err")
    ("$C" --emit=exe --opt=O2 --xpu-backend="$BE" \
      --classpath="$d/llama.cja,$cp_deps" -o "$d/wgrtiming" \
      dev.cajeta.wgrtime.WgrTiming.run "$d/src" "$b" >/dev/null 2>"$d/exe.err")
    echo "built $arm"
  done
fi
res=$out/results.tsv
: > "$res"
rounds=${ROUNDS:-4}
r=0
while [ "$r" -lt "$rounds" ]; do
  if [ $((r % 2)) -eq 0 ]; then order="hand moved"; else order="moved hand"; fi
  for arm in $order; do
    "$out/$arm/wgrtiming" | sed "s/^/$arm\t/" | tee -a "$out/raw.log" | grep -v "^$arm.#" >> "$res"
  done
  r=$((r + 1))
done
# Per kernel: min of the per-run minima and median of the per-run medians, then moved/hand.
sort -t$'\t' -k2,2 -k1,1 "$res" | awk -F'\t' '
  { key=$2 "\t" $1; n[key]++; if (!(key in mn) || $3 < mn[key]) mn[key]=$3; med[key, n[key]]=$4; ks[$2]=1 }
  END {
    printf "kernel\thand_min_us\tmoved_min_us\tmin_ratio\thand_med_us\tmoved_med_us\tmed_ratio\n"
    for (k in ks) {
      for (a = 0; a < 2; a++) {
        arm = (a == 0) ? "hand" : "moved"; key = k "\t" arm; c = n[key]
        for (i = 1; i <= c; i++) v[i] = med[key, i]
        for (i = 2; i <= c; i++) { x = v[i]; j = i - 1; while (j >= 1 && v[j] > x) { v[j + 1] = v[j]; j-- } v[j + 1] = x }
        m[arm] = (c % 2) ? v[(c + 1) / 2] : (v[c / 2] + v[c / 2 + 1]) / 2
      }
      printf "%s\t%.3f\t%.3f\t%.4f\t%.3f\t%.3f\t%.4f\n", k, mn[k "\thand"], mn[k "\tmoved"],
        mn[k "\tmoved"] / mn[k "\thand"], m["hand"], m["moved"], m["moved"] / m["hand"]
    }
  }' | tee "$out/summary.tsv"
