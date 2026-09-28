#!/usr/bin/env bash
# run-with-watchdog.sh LOG CMD [ARGS...]
#
# Run CMD, stream its merged output to the terminal and to LOG, and stop it if
# the output goes quiet for too long. Exit status is CMD's own, or 124 when it
# was stopped as hung.
#
# WHY. On 2026-09-27 one cajeta-llm test ran for THREE HOURS pinning all 32
# cores and 25 GB while the log did not move — a scaffold loop whose "any
# work-item still active" flag never cleared. run-tests.sh runs one monolithic
# binary with no per-test timeout, so a runaway looked exactly like a slow
# test, and nobody could tell which test it was, because a test prints only
# when it FINISHES.
#
# A log that stops growing IS a stalled test. After HANG_SECONDS of silence
# (default HANG_MINUTES=20, longer than any single test has measured) this
# attaches the debugger to the stalled process and records every thread's
# backtrace into LOG — that names the running test method and shows where it
# is stuck — then kills THAT PROCESS ONLY, by its recorded pid. Never by
# pattern: the CI runners share this box and run the same binary names.
# HANG_SECONDS=0 disables the watchdog.
set -u
log="$1"; shift
hang_s="${HANG_SECONDS:-$(( ${HANG_MINUTES:-20} * 60 ))}"
poll=60
[ "$hang_s" -lt 240 ] && poll=2
: > "$log"
# Through allow-debugger-attach.py: under ptrace_scope=1 only an ancestor may
# attach, and the watchdog below is CMD's sibling, so without the grant its
# backtrace is "Could not attach to process" (measured 2026-09-27).
attach=""
command -v python3 >/dev/null 2>&1 && attach="python3 $(dirname "$0")/allow-debugger-attach.py"
$attach stdbuf -oL -eL "$@" >> "$log" 2>&1 &
pid=$!
wd=""
if [ "$hang_s" -gt 0 ]; then
    (
        while kill -0 "$pid" 2>/dev/null; do
            sleep "$poll"
            kill -0 "$pid" 2>/dev/null || exit 0
            age=$(( $(date +%s) - $(stat -c %Y "$log") ))
            if [ "$age" -ge "$hang_s" ]; then
                {
                    echo ""
                    echo ">> HUNG: no output for ${age}s (limit ${hang_s}s)."
                    echo ">> The last result line above is the last test that FINISHED;"
                    echo ">> the stalled one is named in the backtrace below."
                    if command -v gdb >/dev/null 2>&1; then
                        # The MAIN thread first and in full: it holds the
                        # test method. gdb lists threads highest first, and
                        # on 2026-09-27 thirty-one identical kernel-pool
                        # stacks filled the old 120-line cap, so the report
                        # named the kernel and never reached the test.
                        timeout 120 gdb -p "$pid" -batch \
                            -ex 'echo >> main thread:\n' -ex 'thread 1' -ex 'bt 40' \
                            -ex 'echo >> every thread, top frames:\n' \
                            -ex 'thread apply all bt 4' 2>&1 \
                            | grep -E '^>> |^Thread |^#|^\[Switching' | head -240
                    fi
                    echo ">> stopped pid $pid, the command under test only."
                } >> "$log"
                # The flag BEFORE the kill. Written after it, the main loop's
                # `wait` returned the moment the process died and reaped this
                # watchdog during its grace sleep, so the flag never landed and
                # a hang reported as a plain SIGTERM (143). The fires-arm of
                # test-run-with-watchdog.sh caught exactly that.
                touch "$log.hung"
                kill "$pid" 2>/dev/null
                sleep 3
                kill -9 "$pid" 2>/dev/null
                exit 0
            fi
        done
    ) &
    wd=$!
fi
tail -n +1 -f --pid="$pid" "$log" &
tl=$!
wait "$pid"
rc=$?
# Let a watchdog that fired finish writing; one that did not is killed.
if [ -n "$wd" ]; then
    [ -f "$log.hung" ] || kill "$wd" 2>/dev/null
    wait "$wd" 2>/dev/null
fi
wait "$tl" 2>/dev/null
if [ -f "$log.hung" ]; then
    rm -f "$log.hung"
    exit 124
fi
exit "$rc"
