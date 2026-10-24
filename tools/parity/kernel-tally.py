#!/usr/bin/env python3
"""Window a cupti-tally CSV (start_ns,end_ns,stream,name) and total it by kernel.

  kernel-tally.py run.csv                      # whole run, plus a 100 ms density map
  kernel-tally.py run.csv --from 300ms --to 1.4s
  kernel-tally.py run.csv --last 1s            # the run's last second (a decode tail)

Times are relative to the first record, like `cajeta profile summary`.
"""
import argparse, collections, sys

def dur(s):
    if s is None: return None
    for suf, m in (("ms", 1e6), ("us", 1e3), ("s", 1e9), ("ns", 1)):
        if s.endswith(suf): return float(s[:-len(suf)]) * m
    return float(s)

ap = argparse.ArgumentParser()
ap.add_argument("csv"); ap.add_argument("--from", dest="frm"); ap.add_argument("--to")
ap.add_argument("--last"); ap.add_argument("--top", type=int, default=40)
ap.add_argument("--bucket", default="100ms")
ap.add_argument("--gaps", action="store_true", help="idle between consecutive kernels in the window")
a = ap.parse_args()

rows = []
with open(a.csv) as f:
    for line in f:
        p = line.rstrip("\n").split(",", 3)
        if len(p) < 4: continue
        rows.append((int(p[0]), int(p[1]), p[2], p[3]))
if not rows: sys.exit("no records")
t0 = min(r[0] for r in rows); t1 = max(r[1] for r in rows)
lo = dur(a.frm) or 0.0
hi = dur(a.to) if a.to else (t1 - t0)
if a.last: lo = (t1 - t0) - dur(a.last); hi = t1 - t0

b = dur(a.bucket); dens = collections.Counter()
for s, e, _, _ in rows: dens[int((s - t0) // b)] += (e - s)
print(f"span {(t1-t0)/1e6:.1f} ms, {len(rows)} records; kernel time per {a.bucket} bucket (ms):")
print(" ".join(f"{dens[i]/1e6:.0f}" for i in range(int((t1 - t0) // b) + 1)))

tot = collections.defaultdict(lambda: [0, 0, 0])
inwin = 0
for s, e, _, n in rows:
    if s - t0 < lo or e - t0 > hi: continue
    inwin += 1; r = tot[n]; r[0] += 1; r[1] += e - s; r[2] = max(r[2], e - s)
allk = sum(r[1] for r in tot.values())
print(f"\nwindow {lo/1e6:.1f} .. {hi/1e6:.1f} ms: {inwin} kernels, {allk/1e6:.3f} ms kernel time")
print(f"{'kernel':56} {'count':>8} {'total_us':>11} {'mean_us':>9} {'share':>6}")
for n, (c, t, m) in sorted(tot.items(), key=lambda kv: -kv[1][1])[: a.top]:
    print(f"{n[:56]:56} {c:8d} {t/1e3:11.1f} {t/1e3/c:9.2f} {100*t/allk:5.1f}%")

if a.gaps:
    win = sorted((s, e, n) for s, e, _, n in rows if s - t0 >= lo and e - t0 <= hi)
    idle = 0; big = []
    for (s0, e0, n0), (s1, e1, n1) in zip(win, win[1:]):
        g = s1 - e0
        if g > 0: idle += g
        if g > 20000: big.append((g, n0, n1))
    span = win[-1][1] - win[0][0] if win else 0
    print(f"\ngaps: window span {span/1e6:.1f} ms, kernel {sum(e-s for s,e,_ in win)/1e6:.1f} ms, idle {idle/1e6:.1f} ms ({100*idle/max(span,1):.0f}%)")
    by = collections.defaultdict(lambda: [0, 0])
    for g, n0, n1 in big: r = by[(n0[:28], n1[:28])]; r[0] += 1; r[1] += g
    print(f"gaps over 20 us by (before, after): {len(big)} gaps, {sum(g for g,_,_ in big)/1e6:.1f} ms")
    for (n0, n1), (c, t) in sorted(by.items(), key=lambda kv: -kv[1][1])[:12]:
        print(f"  {c:6d} {t/1e3:9.1f} us  {n0:28} -> {n1}")
