"""Interactive run: checkbox model selection and verbose benchmark with a modern console UI."""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

from typing import Optional

from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from .api_client import OllamaClient
from .benchmark import (
    ModelResult,
    run_single_benchmark,
    _consume_stream_into_holder,
    _model_result_from_stream_holder,
)
from .prune import prune_models
from .system_info import collect_environment_config

# Default prompt for interactive run
# TODO(future): Some models are vision/image models; support a set of images or files to use with models.
DEFAULT_RUN_PROMPT = "write hello world in lisp"

# Use a single console for consistent styling
console = Console()


def _format_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "—"
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(round(seconds)), 60)
    return f"{m}m {s}s"


def _ask_prompt_interactive(default: str = DEFAULT_RUN_PROMPT) -> str:
    """Ask user for the test prompt (when not provided on command line)."""
    try:
        import questionary
    except ImportError:
        console.print(
            "[red]Missing dependency: install with[/] [bold]pip install questionary[/]"
        )
        raise SystemExit(1)
    result = questionary.text(
        "Test prompt to run for each model:",
        default=default,
        style=questionary.Style([
            ("qmark", "fg:green bold"),
            ("question", "bold"),
            ("answer", "fg:cyan"),
        ]),
    ).ask()
    return (result or default).strip() or default


def _select_models_interactive(model_names: list[str]) -> list[str]:
    """Show a checkbox prompt and return the list of selected model names."""
    try:
        import questionary
    except ImportError:
        console.print(
            "[red]Missing dependency: install with[/] [bold]pip install questionary[/]"
        )
        raise SystemExit(1)

    if not model_names:
        return []

    choices = [questionary.Choice(title=name, value=name) for name in model_names]
    prompt = questionary.checkbox(
        "Select models to run (space to toggle, enter to confirm):",
        choices=choices,
        style=questionary.Style([
            ("selected", "fg:cyan bold"),
            ("pointer", "fg:cyan bold"),
            ("highlighted", "fg:cyan"),
            ("qmark", "fg:green bold"),
        ]),
    )
    selected = prompt.ask()
    return selected or []


def _offer_delete_failed(
    client: OllamaClient,
    results: list[ModelResult],
) -> None:
    """If there are failed results, offer a checkbox to select which to delete."""
    failed = [r for r in results if not r.success]
    if not failed:
        return
    try:
        import questionary
    except ImportError:
        return
    choices = [questionary.Choice(title=r.name, value=r.name) for r in failed]
    prompt = questionary.checkbox(
        "Delete failed model(s)? Select which to remove (space to toggle, enter to confirm, none to skip):",
        choices=choices,
        style=questionary.Style([
            ("selected", "fg:red bold"),
            ("pointer", "fg:cyan bold"),
            ("highlighted", "fg:cyan"),
            ("qmark", "fg:green bold"),
        ]),
    )
    to_delete = prompt.ask()
    if not to_delete:
        return
    console.print()
    outcomes = prune_models(client, to_delete)
    for name, success, err in outcomes:
        if success:
            console.print(f"  [green]Deleted:[/] {name}")
        else:
            console.print(f"  [red]Failed to delete {name}:[/] {err or 'Unknown error'}")
    console.print()


def _ask_run_again() -> bool:
    """Ask if the user would like to run again. Returns True to go again."""
    try:
        import questionary
    except ImportError:
        return False
    result = questionary.confirm(
        "Would you like to run again?",
        default=False,
        style=questionary.Style([
            ("qmark", "fg:green bold"),
            ("question", "bold"),
            ("answered_question", "bold"),
            ("answer", "fg:cyan"),
        ]),
    ).ask()
    return result is True


def _model_output_renderable(
    raw_text: str,
    max_lines: int = 20,
) -> Markdown:
    """Detect and render model output as Markdown: code blocks, **bold**, *italic*, lists, etc."""
    if not raw_text:
        return Markdown("")
    lines = raw_text.splitlines()
    if len(lines) > max_lines:
        excerpt = "\n".join(lines[-max_lines:])
        return Markdown("…\n\n" + excerpt)
    return Markdown(raw_text)


def _live_status_component(
    model_name: str,
    index: int,
    total: int,
    elapsed: float,
    done: bool,
    result: Optional[ModelResult] = None,
    streaming_text: Optional[str] = None,
    pause_remaining: Optional[int] = None,
    delay_paused: bool = False,
    display_paused: bool = False,
    paused_snapshot: Optional[str] = None,
) -> Panel:
    """Build the Rich content for the live 'running' panel."""
    header = Text()
    header.append("Running model ", style="dim")
    header.append(model_name, style="bold cyan")
    header.append(f"  ({index}/{total})", style="dim")

    body_parts: list = []
    if done and result is not None:
        if result.success:
            status_line = Text()
            status_line.append("✓ Completed ", style="bold green")
            status_line.append(f"in {_format_duration(result.elapsed_seconds)}", style="green")
            if result.tokens_per_second is not None:
                status_line.append(f"  •  {result.tokens_per_second:.1f} tok/s", style="dim")
            if result.tokens_generated:
                status_line.append(f"  •  {result.tokens_generated} tokens", style="dim")
            status_line.append("\n\n", style="dim")
            body_parts.append(status_line)
            # Keep response visible with formatting preserved (no strip); render as Markdown
            if result.response_text and result.response_text.strip():
                body_parts.append(_model_output_renderable(result.response_text))
        else:
            fail_line = Text()
            fail_line.append("✗ Failed: ", style="bold red")
            fail_line.append(result.error or "Unknown error", style="red")
            body_parts.append(fail_line)
        if pause_remaining is not None and pause_remaining > 0:
            pause_line = Text()
            pause_line.append("\n\n", style="dim")
            if delay_paused:
                pause_line.append("Paused. ", style="bold yellow")
                pause_line.append("Pausing before next model… ", style="dim")
                pause_line.append(f"{pause_remaining}", style="bold yellow")
                pause_line.append(" s  ", style="dim")
                pause_line.append("(p = resume)", style="dim")
            else:
                pause_line.append("Pausing before next model… ", style="dim")
                pause_line.append(f"{pause_remaining}", style="bold yellow")
                pause_line.append(" s", style="dim")
            pause_line.append("\n", style="dim")
            pause_line.append("n = skip delay   p = pause", style="dim")
            body_parts.append(pause_line)
    else:
        # During run: show streaming output (or paused snapshot), then elapsed and hint
        if display_paused and paused_snapshot is not None:
            body_parts.append(Text("Paused — p to resume\n\n", style="bold yellow"))
            body_parts.append(_model_output_renderable(paused_snapshot))
        elif streaming_text and streaming_text.strip():
            # Live model output: preserve formatting, render as Markdown (last N lines)
            body_parts.append(_model_output_renderable(streaming_text))
            if not streaming_text.endswith("\n"):
                body_parts.append(Text("▌", style="bold cyan"))
        else:
            wait_text = Text()
            wait_text.append("Sending prompt and waiting for response…\n", style="dim")
            wait_text.append(
                "Large models can take several minutes. The app is working; please wait.",
                style="italic yellow",
            )
            body_parts.append(wait_text)
        elapsed_text = Text()
        elapsed_text.append(f"\n\nElapsed: {elapsed:.0f}s", style="bold")
        elapsed_text.append("\n", style="dim")
        elapsed_text.append("n = next model   p = pause", style="dim")
        body_parts.append(elapsed_text)

    body = Group(*body_parts) if len(body_parts) > 1 else (body_parts[0] if body_parts else Text())

    return Panel(
        body,
        title=header,
        border_style="cyan" if not done else ("green" if result and result.success else "red"),
        box=box.ROUNDED,
        padding=(0, 1),
    )


def _run_key_reader_thread(holder: dict) -> None:
    """Background thread during model run: read keys and set holder['skip_to_next'] or holder['display_paused']."""
    if sys.platform == "win32":
        try:
            import msvcrt
        except ImportError:
            return
        while holder.get("run_active", True):
            if msvcrt.kbhit():
                try:
                    c = msvcrt.getch().decode("utf-8", errors="ignore")
                    if c in "nN":
                        holder["skip_to_next"] = True
                    elif c in "pP":
                        holder["display_paused"] = not holder.get("display_paused", False)
                        if holder.get("display_paused"):
                            holder["paused_snapshot"] = holder.get("response_text", "")
                except Exception:
                    pass
            time.sleep(0.1)
        return
    try:
        import select
        import termios
        import tty
    except ImportError:
        return
    fd = sys.stdin.fileno()
    try:
        old = termios.tcgetattr(fd)
    except Exception:
        return
    try:
        tty.setcbreak(fd)
        while holder.get("run_active", True):
            r, _, _ = select.select([sys.stdin], [], [], 0.25)
            if r and sys.stdin in r:
                try:
                    c = sys.stdin.read(1)
                    if c in "nN":
                        holder["skip_to_next"] = True
                    elif c in "pP":
                        holder["display_paused"] = not holder.get("display_paused", False)
                        if holder.get("display_paused"):
                            holder["paused_snapshot"] = holder.get("response_text", "")
                except Exception:
                    pass
    finally:
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        except Exception:
            pass


def _delay_key_reader_thread(holder: dict) -> None:
    """Background thread: read single keys and set holder['skip_delay'] or holder['paused']."""
    if sys.platform == "win32":
        try:
            import msvcrt
        except ImportError:
            return
        while holder.get("countdown_active", True):
            if msvcrt.kbhit():
                try:
                    c = msvcrt.getch().decode("utf-8", errors="ignore")
                    if c in "nN":
                        holder["skip_delay"] = True
                    elif c in "pP":
                        holder["paused"] = not holder.get("paused", False)
                except Exception:
                    pass
            time.sleep(0.1)
        return
    # Unix: use select + termios for non-blocking single-key read
    try:
        import select
        import termios
        import tty
    except ImportError:
        return
    fd = sys.stdin.fileno()
    try:
        old = termios.tcgetattr(fd)
    except Exception:
        return
    try:
        tty.setcbreak(fd)
        while holder.get("countdown_active", True):
            r, _, _ = select.select([sys.stdin], [], [], 0.25)
            if r and sys.stdin in r:
                try:
                    c = sys.stdin.read(1)
                    if c in "nN":
                        holder["skip_delay"] = True
                    elif c in "pP":
                        holder["paused"] = not holder.get("paused", False)
                except Exception:
                    pass
    finally:
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        except Exception:
            pass


def _pause_countdown(seconds: int, live: bool) -> None:
    """Show a countdown indicating delay before next model. seconds must be >= 0."""
    if seconds <= 0:
        return
    if live:
        with Live(
            Panel(Text(), title="[dim]Delay[/]", border_style="yellow", box=box.ROUNDED, padding=(0, 1)),
            console=console,
            refresh_per_second=2,
        ) as live_ctx:
            for remaining in range(seconds, 0, -1):
                msg = Text()
                msg.append("Pausing before next model… ", style="dim")
                msg.append(f"{remaining}", style="bold yellow")
                msg.append(" s", style="dim")
                panel = Panel(
                    msg,
                    title="[dim]Delay[/]",
                    border_style="yellow",
                    box=box.ROUNDED,
                    padding=(0, 1),
                )
                live_ctx.update(panel)
                time.sleep(1)
    else:
        console.print(f"[dim]Pausing {seconds}s before next model…[/]")
        for remaining in range(seconds, 0, -1):
            console.print(f"  [dim]{remaining}s remaining[/]")
            time.sleep(1)


def _env_config_to_markdown_lines(env_config: dict) -> list[str]:
    """Format environment config as markdown lines for the report."""
    lines = ["## Environment", ""]
    cpu = env_config.get("cpu", {})
    lines.append("### CPU")
    lines.append(f"- Physical cores: {cpu.get('physical_cores', 'N/A')}")
    lines.append(f"- Logical cores: {cpu.get('logical_cores', 'N/A')}")
    lines.append(f"- Frequency (MHz): {cpu.get('freq_mhz', 'N/A')}")
    lines.append(f"- Brand: {cpu.get('brand', 'N/A') or 'N/A'}")
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
    return lines


def _print_environment_panel(env_config: dict) -> None:
    """Print environment (CPU, memory, GPU, platform) as a Rich panel."""
    parts: list[str] = []
    cpu = env_config.get("cpu", {})
    parts.append(
        f"[bold]CPU:[/] {cpu.get('brand', 'N/A') or 'N/A'}  "
        f"({cpu.get('physical_cores', '?')} physical / {cpu.get('logical_cores', '?')} logical, "
        f"{cpu.get('freq_mhz', 'N/A')} MHz)"
    )
    mem = env_config.get("memory_gb", {})
    parts.append(
        f"[bold]Memory:[/] {mem.get('total', 'N/A')} GB total, "
        f"{mem.get('available', 'N/A')} GB available ({mem.get('percent_used', 'N/A')}% used)"
    )
    gpus = env_config.get("gpus", [])
    if gpus:
        gpu_strs = []
        for i, g in enumerate(gpus):
            gpu_strs.append(
                f"GPU {i}: {g.get('name', 'N/A')} "
                f"({g.get('memory_total_mb', 'N/A')} MB total, {g.get('memory_used_mb', 'N/A')} MB used, "
                f"driver {g.get('driver_version', 'N/A')})"
            )
        parts.append("[bold]GPU(s):[/] " + "  |  ".join(gpu_strs))
    else:
        parts.append("[bold]GPU(s):[/] No NVIDIA GPU or nvidia-smi not available")
    plat = env_config.get("platform", {})
    parts.append(
        f"[bold]Platform:[/] {plat.get('system', 'N/A')} / {plat.get('machine', 'N/A')}  "
        f"(Python {plat.get('python', 'N/A')})"
    )
    console.print(
        Panel(
            "\n".join(parts),
            title="[dim]Environment[/]",
            border_style="dim",
            box=box.ROUNDED,
        )
    )


def _write_run_report(
    results: list[ModelResult],
    prompt_used: str,
    path: str | Path,
    env_config: Optional[dict] = None,
) -> None:
    """Write a single markdown report: env (if provided), prompt, then each model with stats and output."""
    lines = ["# Ollama Model Manager – Run Report", ""]
    if env_config:
        lines.extend(_env_config_to_markdown_lines(env_config))
    lines.append("## Prompt")
    lines.append("")
    lines.append(prompt_used)
    lines.append("")
    for r in results:
        lines.append("---")
        lines.append("")
        lines.append(f"## Model: {r.name}")
        lines.append("")
        lines.append("| Stat | Value |")
        lines.append("|------|-------|")
        status = "✓ OK" if r.success else f"✗ Failed: {r.error or 'Unknown'}"
        if r.success and getattr(r, "cut_short_by_user", False):
            status = "✓ OK (cut short by user)"
        lines.append(f"| Status | {status} |")
        lines.append(f"| Time | {_format_duration(r.elapsed_seconds)} |")
        tokens = str(r.tokens_generated) if r.tokens_generated else "—"
        lines.append(f"| Tokens | {tokens} |")
        speed = f"{r.tokens_per_second:.1f} tok/s" if r.tokens_per_second else "—"
        lines.append(f"| Speed | {speed} |")
        lines.append("")
        lines.append("### Output")
        lines.append("")
        if getattr(r, "cut_short_by_user", False):
            lines.append("*Output was cut short by user (skipped to next model).*")
            lines.append("")
        if r.success and r.response_text not in (None, ""):
            # Write output as raw markdown so report viewers render it (code blocks, bold, etc.)
            lines.append(r.response_text.rstrip())
            lines.append("")
        elif not r.success and r.error:
            lines.append(f"*Error:* `{r.error}`")
        else:
            lines.append("*No output captured.*")
        lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def run_interactive_benchmark(
    client: OllamaClient,
    model_names: list[str],
    prompt: str = DEFAULT_RUN_PROMPT,
    live: bool = True,
    pause_seconds: float = 15,
) -> list[ModelResult]:
    """
    Run benchmarks for the given models one at a time.
    If live=True, show a verbose live console UI; if False, minimal progress lines.
    Pause between models with a countdown (default 15s); set pause_seconds to 0 to skip.
    Returns the list of ModelResult in order.
    """
    results: list[ModelResult] = []
    total = len(model_names)

    for i, model_name in enumerate(model_names, start=1):
        start_time = time.perf_counter()
        result_holder: list[ModelResult] = []
        stream_holder: dict = {}

        def run_in_thread_streaming():
            _consume_stream_into_holder(client, model_name, prompt, stream_holder)

        def run_in_thread_non_streaming():
            r = run_single_benchmark(client, model_name, prompt=prompt)
            result_holder.append(r)

        if live:
            stream_holder["skip_to_next"] = False
            stream_holder["display_paused"] = False
            stream_holder["run_active"] = True
            stream_holder["paused_snapshot"] = None
            thread = threading.Thread(target=run_in_thread_streaming, daemon=True)
            thread.start()
            key_thread = threading.Thread(
                target=_run_key_reader_thread,
                args=(stream_holder,),
                daemon=True,
            )
            key_thread.start()
            with Live(
                _live_status_component(model_name, i, total, 0.0, False),
                console=console,
                refresh_per_second=8,
            ) as live_ctx:
                while not stream_holder.get("done", False):
                    if stream_holder.get("skip_to_next"):
                        break
                    elapsed = time.perf_counter() - start_time
                    if stream_holder.get("display_paused"):
                        live_ctx.update(
                            _live_status_component(
                                model_name,
                                i,
                                total,
                                elapsed,
                                False,
                                streaming_text=stream_holder.get("response_text") or "",
                                display_paused=True,
                                paused_snapshot=stream_holder.get("paused_snapshot") or "",
                            )
                        )
                        time.sleep(0.2)
                    else:
                        live_ctx.update(
                            _live_status_component(
                                model_name,
                                i,
                                total,
                                elapsed,
                                False,
                                streaming_text=stream_holder.get("response_text") or "",
                            )
                        )
                        time.sleep(0.125)
                stream_holder["run_active"] = False
                key_thread.join(timeout=1.0)
                thread.join(timeout=2.0)
                r = _model_result_from_stream_holder(model_name, stream_holder)
                results.append(r)
                elapsed = time.perf_counter() - start_time
                live_ctx.update(
                    _live_status_component(model_name, i, total, elapsed, True, r)
                )
                time.sleep(1.2)
                # Pause with result still visible; n=skip delay, p=pause
                if i < total and pause_seconds > 0:
                    delay_holder = {
                        "skip_delay": False,
                        "paused": False,
                        "countdown_active": True,
                    }
                    key_thread = threading.Thread(
                        target=_delay_key_reader_thread,
                        args=(delay_holder,),
                        daemon=True,
                    )
                    key_thread.start()
                    remaining = int(pause_seconds)
                    try:
                        while remaining > 0 and not delay_holder["skip_delay"]:
                            live_ctx.update(
                                _live_status_component(
                                    model_name,
                                    i,
                                    total,
                                    elapsed,
                                    True,
                                    r,
                                    pause_remaining=remaining,
                                    delay_paused=delay_holder["paused"],
                                )
                            )
                            if delay_holder["paused"]:
                                time.sleep(0.2)
                                continue
                            for _ in range(5):
                                if delay_holder["skip_delay"]:
                                    break
                                time.sleep(0.2)
                            remaining -= 1
                    finally:
                        delay_holder["countdown_active"] = False
                        key_thread.join(timeout=1.0)
        else:
            thread = threading.Thread(target=run_in_thread_non_streaming, daemon=True)
            thread.start()
            console.print(f"[dim]Running {model_name} ({i}/{total})…[/]")
            thread.join(timeout=300)
            if result_holder:
                r = result_holder[0]
                results.append(r)
                if r.success:
                    console.print(
                        f"  [green]✓[/] {_format_duration(r.elapsed_seconds)}"
                        + (f"  {r.tokens_per_second:.1f} tok/s" if r.tokens_per_second else "")
                    )
                else:
                    console.print(f"  [red]✗[/] {r.error}")
            else:
                results.append(
                    ModelResult(
                        name=model_name,
                        success=False,
                        error="No result captured",
                        elapsed_seconds=time.perf_counter() - start_time,
                    )
                )
                console.print("  [red]✗[/] No result captured")
            if i < total and pause_seconds > 0:
                _pause_countdown(int(pause_seconds), live=False)

    return results


def print_results_table(
    results: list[ModelResult],
    prompt_used: str,
    run_elapsed_seconds: Optional[float] = None,
) -> None:
    """Print a Rich table summarizing all benchmark results."""
    table = Table(
        title="Benchmark results",
        box=box.ROUNDED,
        header_style="bold cyan",
        border_style="blue",
        show_lines=False,
        expand=False,
    )
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("Time", justify="right", no_wrap=True)
    table.add_column("Tokens", justify="right", no_wrap=True)
    table.add_column("Speed", justify="right", no_wrap=True)

    # Sort by status (passed first), then speed (desc), then tokens (desc)
    sorted_results = sorted(
        results,
        key=lambda r: (
            0 if r.success else 1,
            -(r.tokens_per_second or 0),
            -r.tokens_generated,
        ),
    )
    for r in sorted_results:
        if r.success:
            status = "[green]✓ OK[/]"
            if getattr(r, "cut_short_by_user", False):
                status = "[green]✓ OK[/] [dim](cut short by user)[/]"
            time_str = _format_duration(r.elapsed_seconds)
            tokens_str = str(r.tokens_generated) if r.tokens_generated else "—"
            speed_str = (
                f"{r.tokens_per_second:.1f} tok/s" if r.tokens_per_second else "—"
            )
        else:
            status = "[red]✗ Failed[/]"
            time_str = _format_duration(r.elapsed_seconds)
            tokens_str = "—"
            speed_str = "—"
        table.add_row(
            r.name,
            status,
            time_str,
            tokens_str,
            speed_str,
        )

    console.print()
    console.print(Panel(
        f"[dim]Prompt:[/] [white]{prompt_used}[/]",
        title="[dim]Test prompt[/]",
        border_style="dim",
        box=box.ROUNDED,
    ))
    console.print()
    console.print(table)

    passed = sum(1 for r in results if r.success)
    failed = len(results) - passed
    console.print()
    if run_elapsed_seconds is not None:
        console.print(
            f"[bold]Overall run time:[/] [cyan]{_format_duration(run_elapsed_seconds)}[/]"
        )
        console.print()
    if failed:
        console.print(
            f"[bold]Summary:[/] [green]{passed} passed[/], [red]{failed} failed[/]"
        )
        for r in results:
            if not r.success and r.error:
                console.print(f"  [red]• {r.name}:[/] {r.error}")
    else:
        console.print(f"[bold green]All {passed} model(s) completed successfully.[/]")


def cmd_run_interactive(
    client: OllamaClient,
    prompt: Optional[str] = None,
    model_list: Optional[list[str]] = None,
    report_path: Optional[str] = None,
    quiet: bool = False,
    pause_seconds: float = 15,
) -> int:
    """
    Interactive flow: optionally ask for prompt, list models, let user select with
    checkboxes (unless model_list is provided), then run benchmark for each
    selected model one at a time with verbose output. At the end, offers to
    delete failed models (checkbox) and to run again.
    """
    initial_prompt = prompt.strip() if prompt and prompt.strip() else None
    initial_model_list = model_list

    while True:
        # Banner (show on first run and when running again)
        console.print()
        console.print(
            Panel(
                "[bold cyan]Ollama Model Manager[/] — [dim]Interactive run[/]",
                box=box.DOUBLE,
                border_style="cyan",
                padding=(0, 2),
            )
        )
        console.print()

        # Fetch models
        console.print("[dim]Fetching model list…[/]")
        ok, err, models = client.list_models()
        if not ok:
            console.print(f"[red]Failed to list models:[/] {err}")
            return 1
        model_names = [m.get("name") for m in models if m.get("name")]
        if not model_names:
            console.print("[yellow]No models found. Pull a model with[/] [bold]ollama pull <name>[/]")
            return 0
        console.print(f"[green]Found {len(model_names)} model(s).[/]")
        console.print()

        # Prompt: use initial if provided, else ask (on "run again" we re-ask unless --prompt was given)
        if initial_prompt is not None:
            prompt = initial_prompt
        elif prompt is None or prompt.strip() == "":
            prompt = _ask_prompt_interactive(DEFAULT_RUN_PROMPT)
            console.print()
        else:
            prompt = prompt.strip()

        if initial_model_list is not None:
            # Non-interactive: use given list (validate against available)
            available = set(model_names)
            selected = [m for m in initial_model_list if m in available]
            missing = [m for m in initial_model_list if m not in available]
            if missing:
                console.print(f"[yellow]Unknown or missing models (skipped):[/] {', '.join(missing)}")
            if not selected:
                console.print("[red]No valid models to run.[/]")
                return 1
        else:
            # Checkbox selection (questionary runs in terminal; do this before heavy Rich Live)
            selected = _select_models_interactive(model_names)
            if not selected:
                console.print("[yellow]No models selected. Exiting.[/]")
                return 0

        console.print()
        console.print(
            Panel(
                f"Prompt: [bold]{prompt}[/]\nModels: [bold]{len(selected)}[/] selected",
                title="[cyan]Run configuration[/]",
                border_style="cyan",
                box=box.ROUNDED,
            )
        )
        console.print()

        # Capture environment (CPU, memory, GPU, platform) for this run
        env_config = collect_environment_config()

        # Run each model one at a time (live UI unless --quiet)
        run_start = time.perf_counter()
        results = run_interactive_benchmark(
            client,
            selected,
            prompt=prompt,
            live=not quiet,
            pause_seconds=pause_seconds,
        )
        run_elapsed = time.perf_counter() - run_start

        # Summary table with overall run time
        print_results_table(results, prompt, run_elapsed_seconds=run_elapsed)

        # Environment (CPU, memory, GPU, platform)
        _print_environment_panel(env_config)

        # Write markdown report if requested
        if report_path:
            _write_run_report(results, prompt, report_path, env_config=env_config)
            console.print(f"[dim]Report written to:[/] [cyan]{report_path}[/]")

        # Offer to delete failed models (checkbox so user can pick which)
        _offer_delete_failed(client, results)

        # Ask to run again
        if not _ask_run_again():
            break

    return 0
