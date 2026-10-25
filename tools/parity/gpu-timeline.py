#!/usr/bin/env python3
"""Attribute GPU idle time to what the host was doing, from a cupti-tally capture.

  CUDA_INJECTION64_PATH=$PWD/libcupti-tally.so CUPTI_TALLY_OUT=k.csv \\
      CUPTI_TALLY_HOST_OUT=h.csv <program>
  gpu-timeline.py k.csv h.csv                     # the whole run
  gpu-timeline.py k.csv h.csv --from 2.1s --to 2.5s

k.csv is the device side (start_ns,end_ns,stream,name: kernels, copies, fills);
h.csv the host side (start_ns,end_ns,thread,correlation,driver call) on the
CPU clock, and h.csv.corr (correlation,kernel start) the pairs that move it
onto the device clock: under WSL the two differ by tens of seconds and drift.
Times on the command line are relative to the first device record, as in
kernel-tally.py.

Every nanosecond of the window where no kernel, copy or fill runs on any stream
is GPU idle. Each idle nanosecond goes to the highest-priority host activity
overlapping it, on any thread:

  sync    a stream, event or context synchronize: the host waited on the device
  alloc   device memory allocated, freed, reserved or mapped
  copy    a copy or fill call (its device time is busy, the call itself is not)
  launch  a kernel or graph launch call: the host could not issue fast enough
  driver  any other driver call (event record, stream wait, attribute queries)
  host    no driver call at all: the program's own code between calls

A large `host` share is time in the engine itself; a large `launch` share is
the launch floor; `sync` while the device is idle is a wait that ended late or
a round trip the step did not need.
"""
import argparse
import collections
import sys

PRIORITY = ["sync", "alloc", "copy", "launch", "driver"]


def dur(s):
    if s is None:
        return None
    for suf, m in (("ms", 1e6), ("us", 1e3), ("s", 1e9), ("ns", 1)):
        if s.endswith(suf):
            return float(s[:-len(suf)]) * m
    return float(s)


def category(api):
    a = api.lower()
    if "synchronize" in a or "eventquery" in a or "streamquery" in a:
        return "sync"
    if ("memalloc" in a or "memfree" in a or "memaddress" in a or "memmap" in a
            or "memunmap" in a or "memcreate" in a or "memrelease" in a
            or "memhostalloc" in a or "memhostregister" in a or "memsetaccess" in a
            or "cudamalloc" in a or "cudafree" in a or "cudahostalloc" in a):
        return "alloc"
    if "memcpy" in a or "memset" in a or "memprefetch" in a:
        return "copy"
    if "launch" in a:
        return "launch"
    return "driver"


def busy_union(spans):
    out = []
    for s, e in sorted(spans):
        if out and s <= out[-1][1]:
            if e > out[-1][1]:
                out[-1] = (out[-1][0], e)
        else:
            out.append((s, e))
    return out


def idle_intervals(busy, lo, hi):
    out = []
    t = lo
    for s, e in busy:
        if e <= lo or s >= hi:
            continue
        if s > t:
            out.append((t, min(s, hi)))
        t = max(t, e)
    if t < hi:
        out.append((t, hi))
    return out


def attribute(idle, host):
    """ns of idle per category; `host` rows are (start, end, thread, api)."""
    by = collections.Counter()
    # Boundaries where the set of active host categories can change.
    evs = []
    for s, e, _, api in host:
        c = category(api)
        evs.append((s, 1, c))
        evs.append((e, -1, c))
    evs.sort(key=lambda x: (x[0], x[1]))
    active = collections.Counter()
    i = 0
    for lo, hi in idle:
        # Advance the sweep to lo.
        while i < len(evs) and evs[i][0] <= lo:
            active[evs[i][2]] += evs[i][1]
            i += 1
        t = lo
        while t < hi:
            nxt = evs[i][0] if i < len(evs) and evs[i][0] < hi else hi
            if nxt > t:
                c = next((p for p in PRIORITY if active[p] > 0), "host")
                by[c] += nxt - t
                t = nxt
            while i < len(evs) and evs[i][0] <= t and t < hi:
                active[evs[i][2]] += evs[i][1]
                i += 1
            if nxt == hi:
                break
    by.setdefault("host", 0)
    return by


def fit_clock(pairs):
    """host ns -> device ns from (host launch start, its first kernel's start)
    pairs. A kernel starts at or after its launch, later when it queues behind
    earlier work, so the fit runs through the lower envelope: the pairs are
    split into time buckets, each contributes its smallest device-minus-host
    delta, and a least-squares line through those points gives the offset and
    the rate (CUPTI's two clocks drift by percents under WSL)."""
    pairs = sorted(pairs)
    if not pairs:
        return lambda t: t
    nb = max(1, min(20, len(pairs) // 3))
    chunks = [pairs[b * len(pairs) // nb:(b + 1) * len(pairs) // nb] for b in range(nb)]
    chunks = [c for c in chunks if c]

    def line(pts):
        if len(pts) == 1:
            off = pts[0][1] - pts[0][0]
            return lambda t: t + off
        n = len(pts)
        mx = sum(p[0] for p in pts) / n
        my = sum(p[1] for p in pts) / n
        sxx = sum((p[0] - mx) ** 2 for p in pts)
        slope = sum((p[0] - mx) * (p[1] - my) for p in pts) / sxx if sxx else 1.0
        return lambda t: my + slope * (t - mx)

    # Within a bucket the drift itself favours the earliest pair, so each pass
    # picks the bucket's lowest point against the previous pass's line.
    f = lambda t: t
    for _ in range(4):
        pts = [min(c, key=lambda p: p[1] - f(p[0])) for c in chunks]
        f = line(pts)
    return f


def read_host(path):
    """Host rows (start, end, thread, api) moved onto the device clock with the
    launch/kernel correlation pairs in `<path>.corr`, when it exists."""
    rows = []
    with open(path) as f:
        for line in f:
            p = line.rstrip("\n").split(",", 4)
            if len(p) < 5:
                continue
            try:
                rows.append((int(p[0]), int(p[1]), p[2], int(p[3]), p[4]))
            except ValueError:
                continue
    first = {}
    try:
        with open(path + ".corr") as f:
            for line in f:
                c, k = line.strip().split(",")
                c = int(c); k = int(k)
                if c not in first or k < first[c]:
                    first[c] = k
    except FileNotFoundError:
        pass
    pairs = [(s, first[c]) for s, e, th, c, api in rows if c and c in first and category(api) == "launch"]
    f = fit_clock(pairs) if pairs else (lambda t: t)
    return [(int(f(s)), int(f(e)), th, api) for s, e, th, c, api in rows], len(pairs)


def read_csv(path, last_is_name=True):
    rows = []
    with open(path) as f:
        for line in f:
            p = line.rstrip("\n").split(",", 3)
            if len(p) < 4:
                continue
            try:
                rows.append((int(p[0]), int(p[1]), p[2], p[3]))
            except ValueError:
                continue
    return rows


def report(kcsv, hcsv, frm_ns, to_ns):
    dev = read_csv(kcsv)
    if not dev:
        sys.exit("no device records")
    t0 = min(r[0] for r in dev)
    t1 = max(r[1] for r in dev)
    lo = t0 + (frm_ns or 0.0)
    hi = t0 + to_ns if to_ns is not None else t1
    clipped = [(max(s, lo), min(e, hi)) for s, e, _, _ in dev if e > lo and s < hi]
    busy = busy_union(clipped)
    idle = idle_intervals(busy, lo, hi)
    hrows, npairs = read_host(hcsv) if hcsv else ([], 0)
    host = [(max(s, lo), min(e, hi), th, api) for s, e, th, api in hrows
            if e > lo and s < hi]
    by = attribute(idle, host)
    calls = collections.defaultdict(lambda: [0, 0])
    for s, e, _, api in host:
        c = calls[api]
        c[0] += 1
        c[1] += e - s
    gaps = sorted(((e - s, s) for s, e in idle), reverse=True)
    return {"t0": t0, "lo": lo, "hi": hi, "busy": sum(e - s for s, e in busy),
            "idle": sum(e - s for s, e in idle), "by": by, "calls": calls, "gaps": gaps,
            "host": host, "pairs": npairs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kernels")
    ap.add_argument("host")
    ap.add_argument("--from", dest="frm")
    ap.add_argument("--to")
    ap.add_argument("--top", type=int, default=12)
    a = ap.parse_args()
    r = report(a.kernels, a.host, dur(a.frm), dur(a.to))
    span = r["hi"] - r["lo"]
    print(f"window {(r['lo']-r['t0'])/1e6:.1f} .. {(r['hi']-r['t0'])/1e6:.1f} ms: "
          f"span {span/1e6:.3f} ms, GPU busy {r['busy']/1e6:.3f} ms ({100*r['busy']/span:.1f}%), "
          f"idle {r['idle']/1e6:.3f} ms; host clock aligned on {r['pairs']} launch/kernel pairs")
    print(f"\n{'idle while the host was in':28} {'ms':>9} {'of idle':>8}")
    for c in PRIORITY + ["host"]:
        v = r["by"].get(c, 0)
        print(f"{c:28} {v/1e6:9.3f} {100*v/max(1, r['idle']):7.1f}%")
    print(f"\n{'driver call':40} {'count':>8} {'total_ms':>10} {'mean_us':>9}")
    for api, (n, t) in sorted(r["calls"].items(), key=lambda kv: -kv[1][1])[: a.top]:
        print(f"{api[:40]:40} {n:8d} {t/1e6:10.3f} {t/1e3/n:9.2f}")
    print(f"\nlargest idle gaps (ms at offset):")
    for g, s in r["gaps"][: a.top]:
        inside = sorted({api for hs, he, _, api in r["host"] if he > s and hs < s + g})
        print(f"  {g/1e6:8.3f} at {(s-r['t0'])/1e6:9.3f}  {', '.join(inside[:6]) or '(no driver call)'}")


if __name__ == "__main__":
    main()
