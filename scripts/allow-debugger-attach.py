#!/usr/bin/env python3
"""allow-debugger-attach.py CMD [ARGS...]

Let any process of this user attach a debugger to CMD, then become CMD.

ptrace_scope=1 (the Ubuntu default, and Phoenix) lets a debugger attach only
to its own descendants. The hang watchdog is CMD's sibling, so its gdb was
refused and a hang would have been stopped with no backtrace, which is the one
thing that names the stalled test. PR_SET_PTRACER_ANY lifts that for this
process, and exec keeps the same process, so the grant carries to CMD.
Where prctl is missing (not Linux, or Yama off) this just execs.
"""
import ctypes
import os
import sys

PR_SET_PTRACER = 0x59616D61
PR_SET_PTRACER_ANY = ctypes.c_ulong(-1)

if len(sys.argv) < 2:
    sys.stderr.write("usage: allow-debugger-attach.py CMD [ARGS...]\n")
    sys.exit(2)
try:
    ctypes.CDLL(None, use_errno=True).prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY, 0, 0, 0)
except (OSError, AttributeError):
    pass
os.execvp(sys.argv[1], sys.argv[1:])
