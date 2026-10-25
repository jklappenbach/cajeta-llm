#!/usr/bin/env python3
"""Tests for gpu-timeline.py: GPU idle attributed to what the host was doing.

    python3 tools/parity/test_gpu_timeline.py

The timelines are synthetic, in nanoseconds, so every expected number can be
read off the fixture by hand.
"""
import importlib.util
import os
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("gpu_timeline", os.path.join(HERE, "gpu-timeline.py"))
gt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gt)


class Category(unittest.TestCase):
    def test_driver_calls_fall_into_their_category(self):
        self.assertEqual(gt.category("cuStreamSynchronize"), "sync")
        self.assertEqual(gt.category("cuEventSynchronize"), "sync")
        self.assertEqual(gt.category("cuCtxSynchronize"), "sync")
        self.assertEqual(gt.category("cuMemAlloc_v2"), "alloc")
        self.assertEqual(gt.category("cuMemFree_v2"), "alloc")
        self.assertEqual(gt.category("cuMemAddressReserve"), "alloc")
        self.assertEqual(gt.category("cuMemcpyHtoDAsync_v2"), "copy")
        self.assertEqual(gt.category("cuMemsetD8Async"), "copy")
        self.assertEqual(gt.category("cuLaunchKernel"), "launch")
        self.assertEqual(gt.category("cuGraphLaunch"), "launch")
        self.assertEqual(gt.category("cuStreamWaitEvent"), "driver")
        self.assertEqual(gt.category("cuEventRecord"), "driver")

    def test_runtime_calls_fall_into_the_same_categories(self):
        self.assertEqual(gt.category("rt:cudaStreamSynchronize_v3020"), "sync")
        self.assertEqual(gt.category("rt:cudaMalloc_v3020"), "alloc")
        self.assertEqual(gt.category("rt:cudaFree_v3020"), "alloc")
        self.assertEqual(gt.category("rt:cudaMallocAsync_v11020"), "alloc")
        self.assertEqual(gt.category("rt:cudaMemcpyAsync_v3020"), "copy")
        self.assertEqual(gt.category("rt:cudaLaunchKernel_v7000"), "launch")
        self.assertEqual(gt.category("rt:cudaGraphLaunch_v10000"), "launch")
        self.assertEqual(gt.category("rt:cudaGetDevice_v3020"), "driver")


class Attribution(unittest.TestCase):
    def test_busy_union_merges_overlapping_streams(self):
        busy = gt.busy_union([(0, 10), (5, 20), (30, 40)])
        self.assertEqual(busy, [(0, 20), (30, 40)])

    def test_idle_is_the_complement_of_busy_inside_the_window(self):
        idle = gt.idle_intervals([(10, 20), (30, 40)], 0, 50)
        self.assertEqual(idle, [(0, 10), (20, 30), (40, 50)])

    def test_each_idle_nanosecond_goes_to_the_highest_priority_host_activity(self):
        # GPU idle 100..200. Host: a launch 100..130, a sync 120..160 on another
        # thread (sync outranks launch where they overlap), an alloc 170..180,
        # nothing 130..? except the sync, and nothing at all 180..200.
        idle = [(100, 200)]
        host = [(100, 130, 1, "cuLaunchKernel"),
                (120, 160, 2, "cuStreamSynchronize"),
                (170, 180, 1, "cuMemAlloc_v2")]
        got = gt.attribute(idle, host)
        self.assertEqual(got["launch"], 20)   # 100..120
        self.assertEqual(got["sync"], 40)     # 120..160
        self.assertEqual(got["alloc"], 10)    # 170..180
        self.assertEqual(got["host"], 30)     # 160..170 and 180..200: no driver call
        self.assertEqual(sum(got.values()), 100)

    def test_host_activity_outside_idle_time_is_not_counted(self):
        got = gt.attribute([(0, 10)], [(20, 30, 1, "cuMemAlloc_v2")])
        self.assertEqual(got.get("alloc", 0), 0)
        self.assertEqual(got["host"], 10)


class ClockAlignment(unittest.TestCase):
    """CUPTI stamps host calls on the CPU clock and kernels on the GPU's; under
    WSL the bases differ by tens of seconds and the rates by a few percent. The
    launch call and its first kernel share a correlation id, so the earliest
    kernel after each launch pins the device time of that host instant."""

    def test_a_pure_offset_is_recovered(self):
        # Device = host + 5000, every kernel starting 3 ns after its launch began
        # except some that queued longer; the fit takes the lower envelope.
        pairs = [(h, h + 5000 + lat) for h, lat in
                 [(0, 3), (100, 3), (200, 40), (300, 3), (400, 90), (500, 3)]]
        f = gt.fit_clock(pairs)
        self.assertAlmostEqual(f(0), 5003, delta=1)
        self.assertAlmostEqual(f(500), 5503, delta=1)

    def test_a_rate_difference_is_recovered(self):
        # Device runs 4.56% fast against the host, offset 1e9.
        pairs = []
        for i in range(200):
            h = i * 50_000_000          # every 50 ms over 10 s
            lat = 2000 if i % 7 else 900_000   # mostly 2 us, sometimes queued
            pairs.append((h, int(1e9 + h * 1.0456) + lat))
        f = gt.fit_clock(pairs)
        for h in (0, 5_000_000_000, 9_950_000_000):
            self.assertAlmostEqual(f(h), 1e9 + h * 1.0456 + 2000, delta=20_000)

    def test_host_rows_with_correlation_ids_are_moved_onto_the_device_clock(self):
        d = tempfile.mkdtemp()
        k = os.path.join(d, "k.csv")
        h = os.path.join(d, "h.csv")
        c = os.path.join(d, "h.csv.corr")
        # Kernels at device 10_000..10_100 and 10_300..10_400.
        with open(k, "w") as f:
            f.write("10000,10100,7,gemm [1x1x1/32]\n")
            f.write("10300,10400,7,gemm [1x1x1/32]\n")
        # Host clock = device - 9000. Launches at host 990 and 1290 (device
        # 9990, 10290: 10 ns before each kernel), an alloc at host 1100..1200.
        with open(h, "w") as f:
            f.write("990,1000,3,11,cuLaunchKernel\n")
            f.write("1100,1200,3,0,cuMemAlloc_v2\n")
            f.write("1290,1300,3,12,cuLaunchKernel\n")
        with open(c, "w") as f:
            f.write("11,10000\n12,10300\n")
        rep = gt.report(k, h, None, None)
        self.assertEqual(rep["idle"], 200)
        self.assertEqual(rep["by"]["alloc"], 100)
        self.assertEqual(rep["by"]["host"], 100)


class EndToEnd(unittest.TestCase):
    def test_the_report_reads_both_csvs_and_windows_them(self):
        d = tempfile.mkdtemp()
        k = os.path.join(d, "k.csv")
        h = os.path.join(d, "h.csv")
        # Kernels at 1000..1100 and 1300..1400 (the first record is t0 = 1000).
        with open(k, "w") as f:
            f.write("1000,1100,7,gemm [1x1x1/32]\n")
            f.write("1300,1400,7,gemm [1x1x1/32]\n")
        # Host: an alloc covering half the 200 ns gap.
        with open(h, "w") as f:
            f.write("1100,1200,3,0,cuMemAlloc_v2\n")
        rep = gt.report(k, h, None, None)
        self.assertEqual(rep["busy"], 200)
        self.assertEqual(rep["idle"], 200)
        self.assertEqual(rep["by"]["alloc"], 100)
        self.assertEqual(rep["by"]["host"], 100)
        # Windowed to 0..150 ns after t0: one kernel, 50 ns idle with the alloc over it.
        rep = gt.report(k, h, 0.0, 150.0)
        self.assertEqual(rep["busy"], 100)
        self.assertEqual(rep["idle"], 50)
        self.assertEqual(rep["by"]["alloc"], 50)


if __name__ == "__main__":
    unittest.main()
