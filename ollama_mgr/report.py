"""Build a text/markdown report: what worked, what failed, env config, nvidia-smi summary."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .benchmark import ModelResult
    from .system_info import NvidiaSmiSample


def format_report(
    results: list["ModelResult"],
    env_config: dict,
    nvidia_smi_samples: list["NvidiaSmiSample"],
) -> str:
    """Produce a single report string (markdown-style) with env, results, and nvidia-smi summary."""
    lines = []
    lines.append("# Ollama Model Manager – Run Report")
    lines.append("")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    lines.append("")
    lines.append("## Environment configuration")
    lines.append("")
    cpu = env_config.get("cpu", {})
    lines.append("### CPU")
    lines.append(f"- Physical cores: {cpu.get('physical_cores', 'N/A')}")
    lines.append(f"- Logical cores: {cpu.get('logical_cores', 'N/A')}")
    lines.append(f"- Frequency (MHz): {cpu.get('freq_mhz', 'N/A')}")
    lines.append(f"- Brand: {cpu.get('brand', 'N/A')}")
    lines.append("")
    mem = env_config.get("memory_gb", {})
    lines.append("### Memory")
    lines.append(f"- Total (GB): {mem.get('total', 'N/A')}")
    lines.append(f"- Available (GB): {mem.get('available', 'N/A')}")
    lines.append(f"- Percent used: {mem.get('percent_used', 'N/A')}%")
    lines.append("")
    gpus = env_config.get("gpus", [])
    lines.append("### GPU(s)")
    if not gpus:
        lines.append("- No GPU info (nvidia-smi not available or no NVIDIA GPUs)")
    else:
        for i, g in enumerate(gpus):
            lines.append(f"- **GPU {i}**: {g.get('name', 'N/A')}")
            lines.append(f"  - Memory total (MB): {g.get('memory_total_mb', 'N/A')}")
            lines.append(f"  - Memory used (MB): {g.get('memory_used_mb', 'N/A')}")
            lines.append(f"  - Driver: {g.get('driver_version', 'N/A')}")
    lines.append("")
    plat = env_config.get("platform", {})
    lines.append("### Platform")
    lines.append(f"- System: {plat.get('system', 'N/A')} / {plat.get('machine', 'N/A')}")
    lines.append(f"- Python: {plat.get('python', 'N/A')}")
    lines.append("")
    lines.append("## nvidia-smi usage during run (sampled every 10s)")
    lines.append("")
    if not nvidia_smi_samples:
        lines.append("No samples collected (run was short or nvidia-smi not available).")
    else:
        for s in nvidia_smi_samples:
            ts = time.strftime("%H:%M:%S", time.localtime(s.timestamp))
            lines.append(f"- **{ts}** – {s.summary}")
        lines.append("")
        lines.append("Raw samples (for reference):")
        for s in nvidia_smi_samples:
            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(s.timestamp))
            lines.append(f"```\n[{ts}]\n{s.raw_output}\n```")
    lines.append("")
    lines.append("## Model benchmark results")
    lines.append("")
    worked = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    lines.append(f"- **Passed**: {len(worked)}")
    lines.append(f"- **Failed**: {len(failed)}")
    lines.append("")
    lines.append("### Passed")
    if not worked:
        lines.append("None.")
    else:
        for r in worked:
            tps = f" {r.tokens_per_second} tok/s" if r.tokens_per_second is not None else ""
            elapsed = f" ({r.elapsed_seconds}s)" if r.elapsed_seconds is not None else ""
            lines.append(f"- **{r.name}**{elapsed}{tps}")
    lines.append("")
    lines.append("### Failed")
    if not failed:
        lines.append("None.")
    else:
        for r in failed:
            err = f": {r.error}" if r.error else ""
            lines.append(f"- **{r.name}**{err}")
    lines.append("")
    return "\n".join(lines)
