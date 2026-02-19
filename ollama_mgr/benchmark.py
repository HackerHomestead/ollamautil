"""Run each model with a test prompt and measure latency and tokens/sec."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from .api_client import OllamaClient
from .system_info import NvidiaSmiSample, NvidiaSmiSampler

# TODO(future): Support a set of images/files for vision/image models.
DEFAULT_TEST_PROMPT = "Say exactly: OK"


@dataclass
class ModelResult:
    name: str
    success: bool
    error: Optional[str] = None
    elapsed_seconds: Optional[float] = None
    eval_duration_seconds: Optional[float] = None  # from API if present
    tokens_generated: int = 0
    tokens_per_second: Optional[float] = None
    response_text: Optional[str] = None  # raw model output for reports
    cut_short_by_user: bool = False  # True if user pressed 'n' to skip to next model


def _consume_stream_into_holder(
    client: OllamaClient,
    model_name: str,
    prompt: str,
    holder: dict,
) -> None:
    """
    Run streaming generate and update holder with response_text, tokens, eval_duration, done, error.
    Used so the main thread can show live output while this runs in a background thread.
    """
    start = time.perf_counter()
    holder["response_text"] = ""
    holder["tokens"] = 0
    holder["eval_duration"] = None
    holder["done"] = False
    holder["error"] = None
    holder["cut_short_by_user"] = False
    try:
        ok, err, _, stream = client.generate(model_name, prompt, stream=True)
        if not ok:
            holder["error"] = err or "Unknown error"
            holder["done"] = True
            return
        if stream is None:
            holder["error"] = "No stream returned"
            holder["done"] = True
            return
        for chunk in stream:
            if holder.get("skip_to_next"):
                holder["cut_short_by_user"] = True
                break
            if isinstance(chunk, dict):
                delta = chunk.get("response")
                if isinstance(delta, str):
                    holder["response_text"] = holder["response_text"] + delta
                if "eval_count" in chunk:
                    try:
                        holder["tokens"] = int(chunk["eval_count"])
                    except (TypeError, ValueError):
                        pass
                if "eval_duration" in chunk:
                    try:
                        ns = chunk["eval_duration"]
                        holder["eval_duration"] = float(ns) / 1e9
                    except (TypeError, ValueError):
                        pass
        holder["done"] = True
    except Exception as e:
        holder["error"] = str(e)
        holder["done"] = True
    finally:
        holder["elapsed_seconds"] = time.perf_counter() - start


def _estimate_tokens_from_text(text: str) -> int:
    """Rough token estimate when API didn't return eval_count (e.g. stream cut short). ~4 chars/token is a common heuristic."""
    if not text or not text.strip():
        return 0
    return max(0, len(text) // 4)


def _model_result_from_stream_holder(model_name: str, holder: dict) -> ModelResult:
    """Build ModelResult from a holder populated by _consume_stream_into_holder."""
    elapsed = holder.get("elapsed_seconds") or 0
    if holder.get("error"):
        return ModelResult(
            name=model_name,
            success=False,
            error=holder["error"],
            elapsed_seconds=round(elapsed, 2),
        )
    eval_duration = holder.get("eval_duration")
    tokens = holder.get("tokens") or 0
    response_text = holder.get("response_text") or ""
    cut_short = holder.get("cut_short_by_user", False)
    # When cut short, API often never sends final chunk with eval_count/eval_duration; keep benchmark by estimating tokens from text
    if cut_short and tokens == 0 and response_text:
        tokens = _estimate_tokens_from_text(response_text)
    tps = None
    if eval_duration and eval_duration > 0 and tokens > 0:
        tps = tokens / eval_duration
    elif elapsed > 0 and tokens > 0:
        tps = tokens / elapsed
    return ModelResult(
        name=model_name,
        success=True,
        elapsed_seconds=round(elapsed, 2),
        eval_duration_seconds=round(eval_duration, 2) if eval_duration else None,
        tokens_generated=tokens,
        tokens_per_second=round(tps, 2) if tps is not None else None,
        response_text=response_text or None,
        cut_short_by_user=cut_short,
    )


def run_single_benchmark(
    client: OllamaClient,
    model_name: str,
    prompt: str = DEFAULT_TEST_PROMPT,
    max_wait_seconds: int = 120,
) -> ModelResult:
    """Run one model with the test prompt; return timing and token count."""
    start = time.perf_counter()
    ok, err, data, _ = client.generate(model_name, prompt, stream=False)
    elapsed = time.perf_counter() - start
    if not ok:
        return ModelResult(name=model_name, success=False, error=err, elapsed_seconds=elapsed)
    eval_duration = None
    response_text = None
    if isinstance(data, dict):
        eval_duration = data.get("eval_duration")
        if eval_duration is not None:
            try:
                eval_duration = float(eval_duration) / 1e9  # often in nanoseconds
            except (TypeError, ValueError):
                pass
        response_text = data.get("response")
        if response_text is not None and not isinstance(response_text, str):
            response_text = str(response_text)
    # Token count: some APIs return eval_count or similar
    tokens = 0
    if isinstance(data, dict):
        tokens = data.get("eval_count", 0) or 0
        if isinstance(tokens, (list, tuple)):
            tokens = len(tokens)
        try:
            tokens = int(tokens)
        except (TypeError, ValueError):
            tokens = 0
    tps = None
    if eval_duration and eval_duration > 0 and tokens > 0:
        tps = tokens / eval_duration
    elif elapsed > 0 and tokens > 0:
        tps = tokens / elapsed
    return ModelResult(
        name=model_name,
        success=True,
        elapsed_seconds=round(elapsed, 2),
        eval_duration_seconds=round(eval_duration, 2) if eval_duration else None,
        tokens_generated=tokens,
        tokens_per_second=round(tps, 2) if tps is not None else None,
        response_text=response_text,
    )


def run_all_benchmarks(
    client: OllamaClient,
    model_names: list[str],
    test_prompt: str = DEFAULT_TEST_PROMPT,
    nvidia_smi_interval: float = 10.0,
) -> tuple[list[ModelResult], list[NvidiaSmiSample], dict]:
    """
    Run benchmark for each model. Starts nvidia-smi sampler every nvidia_smi_interval seconds,
    runs each model, then stops sampler. Returns (results, nvidia_smi_samples, env_config).
    """
    from .system_info import NvidiaSmiSampler, collect_environment_config

    env_config = collect_environment_config()
    sampler = NvidiaSmiSampler(interval_seconds=nvidia_smi_interval)
    sampler.start()
    results: list[ModelResult] = []
    try:
        for name in model_names:
            results.append(
                run_single_benchmark(client, name, prompt=test_prompt)
            )
    finally:
        sampler.stop()
    samples = sampler.get_samples()
    return results, samples, env_config
