#!/usr/bin/env bash
# Both arms of the hang watchdog, because a check needs a test that it FIRES
# and a test that it does NOT (CLAUDE.md §5): a watchdog that never fires reads
# exactly like a suite that never hangs.
set -u
here="$(cd "$(dirname "$0")" && pwd)"
w="$here/run-with-watchdog.sh"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
fail=0

# 1. FIRES: prints once, then goes silent forever.
HANG_SECONDS=6 "$w" "$tmp/a.log" bash -c 'echo "  PASS first"; sleep 600' >/dev/null 2>&1
rc=$?
if [ "$rc" -ne 124 ]; then echo "FAIL fires: exit $rc, want 124"; fail=1; fi
grep -q '^>> HUNG:' "$tmp/a.log" || { echo "FAIL fires: no HUNG line"; fail=1; }
grep -q 'PASS first' "$tmp/a.log" || { echo "FAIL fires: output before the hang was lost"; fail=1; }
# The backtrace is the point of firing: it names the stalled test. With
# ptrace_scope=1 (the Ubuntu default, and Phoenix) a debugger may attach only
# to its own descendants, so a watchdog that is not the target's ancestor
# records NOTHING and still passes every other check here. Measured
# 2026-09-27: "Could not attach to process". Require a real frame.
grep -qE '^#[0-9]+ ' "$tmp/a.log" || { echo "FAIL fires: no backtrace frame (ptrace refused?)"; fail=1; }

# 2. DOES NOT FIRE: keeps printing more often than the limit, then exits 0.
HANG_SECONDS=6 "$w" "$tmp/b.log" bash -c 'for i in 1 2 3 4 5; do echo "  PASS $i"; sleep 2; done' >/dev/null 2>&1
rc=$?
if [ "$rc" -ne 0 ]; then echo "FAIL quiet: exit $rc, want 0"; fail=1; fi
grep -q '^>> HUNG:' "$tmp/b.log" && { echo "FAIL quiet: fired on a live suite"; fail=1; }
grep -c 'PASS' "$tmp/b.log" | grep -qx 5 || { echo "FAIL quiet: output lost"; fail=1; }

# 3. PASSES THE EXIT STATUS THROUGH: a suite that fails must still fail.
HANG_SECONDS=6 "$w" "$tmp/c.log" bash -c 'echo "  FAIL x"; exit 3' >/dev/null 2>&1
rc=$?
if [ "$rc" -ne 3 ]; then echo "FAIL status: exit $rc, want 3"; fail=1; fi

# 4. KILLS ONLY ITS OWN PID: an unrelated sleeper survives the hang kill.
sleep 300 & bystander=$!
HANG_SECONDS=4 "$w" "$tmp/d.log" bash -c 'sleep 600' >/dev/null 2>&1
kill -0 "$bystander" 2>/dev/null || { echo "FAIL scope: killed an unrelated process"; fail=1; }
kill "$bystander" 2>/dev/null

[ "$fail" -eq 0 ] && echo "watchdog: 4/4 PASS (fires arm includes a real backtrace)"
exit "$fail"
