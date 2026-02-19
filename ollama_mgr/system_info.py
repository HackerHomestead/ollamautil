"""Capture environment: CPU, memory, GPU(s), and nvidia-smi samples every N seconds."""

from __future__ import annotations

import platform
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional


def nvidia_smi_available() -> bool:
    """Return True if nvidia-smi is present on the system (on PATH)."""
    return shutil.which("nvidia-smi") is not None


@dataclass
class CpuInfo:
    physical_cores: int
    logical_cores: int
    freq_mhz: Optional[float]
    brand: str = ""


@dataclass
class MemoryInfo:
    total_gb: float
    available_gb: float
    percent_used: float


@dataclass
class GpuInfo:
    name: str
    memory_total_mb: Optional[float]
    memory_used_mb: Optional[float]
    driver_version: str = ""


@dataclass
class NvidiaSmiSample:
    timestamp: float
    raw_output: str
    summary: str  # one-line summary per GPU if parseable


def get_cpu_info() -> CpuInfo:
    try:
        import psutil
        physical = psutil.cpu_count(logical=False) or 0
        logical = psutil.cpu_count(logical=True) or 0
        freq = psutil.cpu_freq()
        freq_mhz = freq.current if freq else None
        # Best-effort brand (Linux: /proc/cpuinfo)
        brand = ""
        try:
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "model name" in line:
                        brand = line.split(":", 1)[1].strip()
                        break
        except Exception:
            brand = platform.processor() or ""
        return CpuInfo(
            physical_cores=physical,
            logical_cores=logical,
            freq_mhz=freq_mhz,
            brand=brand,
        )
    except Exception:
        return CpuInfo(physical_cores=0, logical_cores=0, freq_mhz=None, brand=platform.processor() or "")


def get_memory_info() -> MemoryInfo:
    try:
        import psutil
        v = psutil.virtual_memory()
        total_gb = v.total / (1024**3)
        available_gb = v.available / (1024**3)
        percent = v.percent
        return MemoryInfo(total_gb=total_gb, available_gb=available_gb, percent_used=percent)
    except Exception:
        return MemoryInfo(total_gb=0.0, available_gb=0.0, percent_used=0.0)


def run_nvidia_smi() -> tuple[str, str]:
    """Run nvidia-smi and return (full stdout+stderr, one-line summary). Only runs if nvidia-smi is on PATH."""
    if not nvidia_smi_available():
        return "nvidia-smi not found (not on PATH)", "nvidia-smi not found"
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        raw = (out.stdout or "") + (out.stderr or "")
        lines = [l.strip() for l in (out.stdout or "").strip().splitlines()]
        summary = " | ".join(lines) if lines else "nvidia-smi: no output"
        return raw, summary
    except FileNotFoundError:
        return "nvidia-smi not found (no NVIDIA GPU or driver)", "nvidia-smi not found"
    except subprocess.TimeoutExpired:
        return "nvidia-smi timed out", "timeout"
    except Exception as e:
        return str(e), str(e)


def get_gpu_info() -> list[GpuInfo]:
    """Get GPU list and memory from nvidia-smi (one-time snapshot). Only runs if nvidia-smi is on PATH."""
    if not nvidia_smi_available():
        return []
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode != 0 or not out.stdout:
            return []
        gpus = []
        for line in out.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                name = parts[0]
                try:
                    mem_total = float(parts[1]) if parts[1] else None
                except ValueError:
                    mem_total = None
                try:
                    mem_used = float(parts[2]) if parts[2] else None
                except ValueError:
                    mem_used = None
                driver = parts[3] if len(parts) > 3 else ""
                gpus.append(GpuInfo(name=name, memory_total_mb=mem_total, memory_used_mb=mem_used, driver_version=driver))
        return gpus
    except Exception:
        return []


class NvidiaSmiSampler:
    """Background thread that runs nvidia-smi every interval_seconds and stores samples."""

    def __init__(self, interval_seconds: float = 10.0):
        self.interval_seconds = interval_seconds
        self.samples: list[NvidiaSmiSample] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self.samples.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self.interval_seconds + 2)
            self._thread = None

    def _run(self) -> None:
        if not nvidia_smi_available():
            return
        while not self._stop.is_set():
            raw, summary = run_nvidia_smi()
            with self._lock:
                self.samples.append(
                    NvidiaSmiSample(timestamp=time.time(), raw_output=raw, summary=summary)
                )
            self._stop.wait(timeout=self.interval_seconds)

    def get_samples(self) -> list[NvidiaSmiSample]:
        with self._lock:
            return list(self.samples)


def collect_environment_config() -> dict:
    """One-shot: CPU, memory, GPU(s), OS."""
    cpu = get_cpu_info()
    mem = get_memory_info()
    gpus = get_gpu_info()
    return {
        "cpu": {
            "physical_cores": cpu.physical_cores,
            "logical_cores": cpu.logical_cores,
            "freq_mhz": cpu.freq_mhz,
            "brand": cpu.brand,
        },
        "memory_gb": {"total": round(mem.total_gb, 2), "available": round(mem.available_gb, 2), "percent_used": round(mem.percent_used, 1)},
        "gpus": [
            {
                "name": g.name,
                "memory_total_mb": g.memory_total_mb,
                "memory_used_mb": g.memory_used_mb,
                "driver_version": g.driver_version,
            }
            for g in gpus
        ],
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
    }
