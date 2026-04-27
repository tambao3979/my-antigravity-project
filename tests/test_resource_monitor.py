import queue
import threading
import time
import unittest

from resource_monitor import (
    ResourceMetricsCollector,
    SystemMonitorAgent,
    put_latest,
)


class FakeVirtualMemory:
    total = 16 * 1024 * 1024 * 1024
    used = 6 * 1024 * 1024 * 1024
    percent = 37.5


class FakePsutil:
    def cpu_percent(self, interval=None):
        if interval is not None:
            raise AssertionError("cpu_percent must not block with an interval")
        return 42.0

    def virtual_memory(self):
        return FakeVirtualMemory()


class FakeMemoryInfo:
    total = 8 * 1024 * 1024 * 1024
    used = 3 * 1024 * 1024 * 1024


class FakeUtilization:
    gpu = 64


class FakePynvml:
    NVML_TEMPERATURE_GPU = 0

    def __init__(self):
        self.init_calls = 0

    def nvmlInit(self):
        self.init_calls += 1

    def nvmlDeviceGetHandleByIndex(self, index):
        return f"gpu-{index}"

    def nvmlDeviceGetName(self, handle):
        return b"RTX Test"

    def nvmlDeviceGetUtilizationRates(self, handle):
        return FakeUtilization()

    def nvmlDeviceGetMemoryInfo(self, handle):
        return FakeMemoryInfo()

    def nvmlDeviceGetTemperature(self, handle, kind):
        return 71


class RaisingPynvml:
    def nvmlInit(self):
        raise RuntimeError("No NVIDIA GPU")


class ResourceMonitorTests(unittest.TestCase):
    def test_collector_reads_cpu_ram_and_gpu_without_blocking_cpu_percent(self):
        nvml = FakePynvml()
        collector = ResourceMetricsCollector(psutil_module=FakePsutil(), pynvml_module=nvml)

        metrics = collector.collect()

        self.assertEqual(metrics.system.cpu_percent, 42.0)
        self.assertEqual(metrics.system.ram_total_mb, 16384.0)
        self.assertEqual(metrics.system.ram_used_mb, 6144.0)
        self.assertEqual(metrics.gpu.name, "RTX Test")
        self.assertEqual(metrics.gpu.utilization_percent, 64.0)
        self.assertEqual(metrics.gpu.vram_used_mb, 3072.0)
        self.assertEqual(metrics.gpu.temperature_c, 71.0)
        self.assertEqual(nvml.init_calls, 1)

    def test_collector_gracefully_falls_back_when_nvml_is_unavailable(self):
        collector = ResourceMetricsCollector(psutil_module=FakePsutil(), pynvml_module=RaisingPynvml())

        metrics = collector.collect()

        self.assertEqual(metrics.system.cpu_percent, 42.0)
        self.assertFalse(metrics.gpu.available)
        self.assertIn("No NVIDIA GPU", metrics.gpu.error)

    def test_put_latest_replaces_stale_queue_item(self):
        q = queue.Queue(maxsize=1)

        put_latest(q, "old")
        put_latest(q, "new")

        self.assertEqual(q.get_nowait(), "new")
        self.assertTrue(q.empty())

    def test_agent_runs_on_background_thread_and_polls_queue(self):
        q = queue.Queue(maxsize=1)
        collector = ResourceMetricsCollector(psutil_module=FakePsutil(), pynvml_module=RaisingPynvml())
        agent = SystemMonitorAgent(output_queue=q, collector=collector, poll_interval=0.01)

        agent.start()
        try:
            deadline = time.time() + 1.0
            metrics = None
            while time.time() < deadline:
                try:
                    metrics = q.get_nowait()
                    break
                except queue.Empty:
                    time.sleep(0.01)
            self.assertIsNotNone(metrics)
            self.assertNotEqual(agent.thread_ident, threading.get_ident())
        finally:
            agent.stop(timeout=1.0)


if __name__ == "__main__":
    unittest.main()
