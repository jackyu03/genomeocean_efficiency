"""
Common metrics utilities for benchmarking.
Includes EnergyMeter (NVML) and FLOPs calculation.
"""
import threading
import time
import logging
import sys

# Optional NVML support
try:
    import pynvml
    NVML_AVAILABLE = True
except Exception:
    NVML_AVAILABLE = False

import os # Need os for env vars

log = logging.getLogger("metrics")

class EnergyMeter:
    def __init__(self, gpu_index: int = 0, interval_s: float = 0.1):
        # Resolve physical GPU index if CUDA_VISIBLE_DEVICES is set
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cvd:
            try:
                # Map logical gpu_index (e.g. 0) to physical index from the list
                # e.g. CVD="2,3", gpu_index=0 -> physical=2
                # e.g. CVD="2,3", gpu_index=1 -> physical=3
                visible_devices = [int(x.strip()) for x in cvd.split(",")]
                if gpu_index < len(visible_devices):
                    self.gpu_index = visible_devices[gpu_index]
                    log.info(f"EnergyMeter: Mapped logical GPU {gpu_index} to physical GPU {self.gpu_index} via CUDA_VISIBLE_DEVICES")
                else:
                    self.gpu_index = gpu_index # Fallback
            except ValueError:
                 self.gpu_index = gpu_index
        else:
            self.gpu_index = gpu_index
            
        self.interval = interval_s
        self.samples = []
        self._stop = threading.Event()
        self._thread = None
        self.kwh = None
        self.ok = NVML_AVAILABLE

    def _runner(self):
        handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
        while not self._stop.is_set():
            try:
                p_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
            except Exception:
                p_mw = None
            
            ts = time.perf_counter()
            # Only append valid non-zero readings
            if p_mw is not None and p_mw > 0:
                self.samples.append((ts, p_mw))
            time.sleep(self.interval)

    def __enter__(self):
        if self.ok:
            try:
                pynvml.nvmlInit()
                self._thread = threading.Thread(target=self._runner, daemon=True)
                self._thread.start()
            except Exception as e:
                log.warning(f"NVML init failed: {e}")
                self.ok = False
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.ok:
            self._stop.set()
            self._thread.join(timeout=1.0)
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
            if len(self.samples) >= 2:
                joules = 0.0
                for (t0, p0_mw), (t1, p1_mw) in zip(self.samples, self.samples[1:]):
                    dt = (t1 - t0)
                    p0 = p0_mw / 1000.0
                    p1 = p1_mw / 1000.0
                    joules += 0.5 * (p0 + p1) * dt
                self.kwh = joules / 3_600_000.0
            else:
                self.kwh = None

def flops_per_seq_decoder(L:int, d:int, T:int, d_ff:int=None) -> float:
    """Estimates FLOPs for a single sequence forward pass in a Decoder-only transformer."""
    if T <= 0 or L <= 0 or d <= 0:
        return 0.0
    m = 4.0 if (d_ff is None) else float(d_ff)/float(d)
    term_proj = (4.0 + 4.0*m) * T * (d**2)
    term_attn = 2.0 * (T**2) * d
    return float(L) * (term_proj + term_attn)
