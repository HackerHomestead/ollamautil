"""Turn-based chat: two models debate/interact with each other in a split-view UI."""

from __future__ import annotations

import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from rich import box

from .api_client import OllamaClient
from .system_info import collect_environment_config

console = Console()

DEFAULT_INITIAL_PROMPT = "Give a short opinion on whether artificial intelligence will help or harm humanity."
DEFAULT_EXCHANGES = 5
DEFAULT_TIMEOUT_SECONDS = 3600


def _format_timestamp(elapsed: float) -> str:
    """Format elapsed seconds as MM:SS."""
    mins, secs = divmod(int(elapsed), 60)
    return f"{mins:02d}:{secs:02d}"


def _select_model_interactive(model_names: list[str], prompt: str) -> Optional[str]:
    """Show a dropdown to select a single model."""
    try:
        import questionary
    except ImportError:
        console.print("[red]Missing dependency: questionary[/]")
        return None

    if not model_names:
        return None

    choices = [questionary.Choice(title=name, value=name) for name in model_names]
    selected = questionary.select(
        prompt,
        choices=choices,
        style=questionary.Style([
            ("selected", "fg:cyan bold"),
            ("pointer", "fg:cyan bold"),
            ("highlighted", "fg:cyan"),
            ("qmark", "fg:green bold"),
        ]),
    ).ask()
    return selected


def _select_two_models(model_names: list[str]) -> tuple[Optional[str], Optional[str]]:
    """Select Player 1 and Player 2 models."""
    player1 = _select_model_interactive(model_names, "Select Player 1 (left panel):")
    if not player1:
        return None, None
    console.print()

    remaining = [m for m in model_names if m != player1]
    player2 = _select_model_interactive(remaining or model_names, "Select Player 2 (right panel):")
    return player1, player2


def _ask_initial_prompt(default: str = DEFAULT_INITIAL_PROMPT) -> str:
    """Ask user for the initial prompt."""
    try:
        import questionary
    except ImportError:
        return default

    result = questionary.text(
        "Initial prompt to start the conversation:",
        default=default,
        style=questionary.Style([
            ("qmark", "fg:green bold"),
            ("question", "bold"),
            ("answer", "fg:cyan"),
        ]),
    ).ask()
    return (result or default).strip() or default


def _ask_summary_model(model_names: list[str]) -> Optional[str]:
    """Ask user if they want a summary and which model to use."""
    try:
        import questionary
    except ImportError:
        return None

    want_summary = questionary.confirm(
        "Would you like a summary of this conversation using another model?",
        default=False,
        style=questionary.Style([
            ("qmark", "fg:green bold"),
            ("question", "bold"),
            ("answered_question", "bold"),
            ("answer", "fg:cyan"),
        ]),
    ).ask()

    if not want_summary:
        return None

    console.print()
    model = _select_model_interactive(model_names, "Select a model to generate the summary:")
    return model


def _ask_summary_prompt() -> str:
    """Ask user for the summary prompt."""
    try:
        import questionary
    except ImportError:
        return "Who won this debate? Provide a summary of each side's arguments."

    result = questionary.text(
        "Summary prompt (e.g., 'Who won this debate?'):",
        default="Who won this debate? Provide a summary of each side's arguments.",
        style=questionary.Style([
            ("qmark", "fg:green bold"),
            ("question", "bold"),
            ("answer", "fg:cyan"),
        ]),
    ).ask()
    return (result or "Who won this debate?").strip()


def _setup_log_file() -> tuple[Path, str]:
    """Create log file in /tmp with timestamp prefix. Returns (path, full_path_str)."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("/tmp")
    log_file = log_dir / f"chat_{timestamp}.md"
    return log_file, str(log_file)


def _init_log_file(path: Path, player1: str, player2: str, initial_prompt: str, start_time: float) -> None:
    """Initialize log file with session info."""
    rel_time = _format_timestamp(0)
    lines = [
        "# Turn-Based Chat Session",
        "",
        f"**Start Time:** {datetime.now().isoformat()}",
        f"**Player 1:** {player1}",
        f"**Player 2:** {player2}",
        "",
        "---",
        "",
        f"## Initial Prompt ({rel_time})",
        "",
        initial_prompt,
        "",
        "---",
        "",
        "## Conversation",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _append_to_log(
    path: Path,
    exchange_num: int,
    player: str,
    content: str,
    elapsed: float,
) -> None:
    """Append a turn to the log file."""
    existing = path.read_text(encoding="utf-8")
    rel_time = _format_timestamp(elapsed)
    new_lines = [
        existing.rstrip(),
        "",
        f"### Exchange {exchange_num} — {player} ({rel_time})",
        "",
        content,
        "",
    ]
    path.write_text("\n".join(new_lines), encoding="utf-8")


def _render_chat_panel(
    title: str,
    content: str,
    border_style: str = "cyan",
    max_lines: int = 30,
) -> Panel:
    """Render a chat panel with Markdown content."""
    if not content:
        return Panel(
            Text("Waiting...", style="dim italic"),
            title=title,
            border_style=border_style,
            box=box.ROUNDED,
            padding=(0, 1),
        )

    lines = content.splitlines()
    if len(lines) > max_lines:
        excerpt = "\n".join(lines[-max_lines:])
        body = Group(
            Text("…\n\n", style="dim"),
            Text(excerpt, style="white"),
        )
    else:
        body = Markdown(content) if len(lines) > 5 else Text(content, style="white")

    return Panel(
        body,
        title=title,
        border_style=border_style,
        box=box.ROUNDED,
        padding=(0, 1),
    )


def _chat_key_reader(holder: dict) -> None:
    """Background thread: read keys for pause/skip/quit/submit."""
    if sys.platform == "win32":
        try:
            import msvcrt
        except ImportError:
            return
        while holder.get("active", True):
            if msvcrt.kbhit():
                try:
                    c = msvcrt.getch().decode("utf-8", errors="ignore")
                    if c in "qQ":
                        holder["quit"] = True
                    elif c in "pP":
                        holder["paused"] = not holder.get("paused", False)
                        if holder.get("paused"):
                            holder["paused_snapshot"] = holder.get("current_player", "")
                    elif c in "sS":
                        holder["skip_next"] = True
                    elif c == "\r" or c == "\n":
                        holder["submit_partial"] = True
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
        while holder.get("active", True):
            r, _, _ = select.select([sys.stdin], [], [], 0.25)
            if r and sys.stdin in r:
                try:
                    c = sys.stdin.read(1)
                    if c in "qQ":
                        holder["quit"] = True
                    elif c in "pP":
                        holder["paused"] = not holder.get("paused", False)
                        if holder.get("paused"):
                            holder["paused_snapshot"] = holder.get("current_player", "")
                    elif c in "sS":
                        holder["skip_next"] = True
                    elif c == "\r" or c == "\n":
                        holder["submit_partial"] = True
                except Exception:
                    pass
    finally:
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        except Exception:
            pass


def _stream_chat_turn(
    client: OllamaClient,
    model: str,
    messages: list[dict],
    holder: dict,
) -> tuple[bool, Optional[str], str]:
    """
    Stream a chat turn. Results stored in holder:
    - holder["response"]: accumulated response text
    - holder["done"]: True when finished
    - holder["error"]: error message if failed
    Returns (ok, error, final_response).
    """
    holder["response"] = ""
    holder["done"] = False
    holder["error"] = None
    holder["tokens_generated"] = 0

    try:
        ok, err, _, stream = client.chat(model, messages, stream=True)
        if not ok or stream is None:
            holder["error"] = err or "Unknown error"
            holder["done"] = True
            return False, err or "Unknown error", ""

        for chunk in stream:
            if holder.get("quit"):
                holder["done"] = True
                break
            if holder.get("skip_next"):
                holder["done"] = True
                holder["skipped"] = True
                break

            content = chunk.get("response", "")
            if content:
                holder["response"] += content

            if "eval_count" in chunk:
                holder["tokens_generated"] = chunk["eval_count"]

        holder["done"] = True
        return True, None, holder.get("response", "")
    except Exception as e:
        holder["error"] = str(e)
        holder["done"] = True
        return False, str(e), ""


def _build_status_panel(
    exchange_num: int,
    current_player: str,
    player1_name: str,
    player2_name: str,
    player1_content: str,
    player2_content: str,
    elapsed: float,
    paused: bool,
    paused_snapshot: Optional[str],
    is_done: bool = False,
    winner: Optional[str] = None,
) -> Group:
    """Build the split-view status panel."""

    header = Text()
    header.append("Turn-Based Chat  ", style="bold cyan")
    header.append(f"Exchange {exchange_num}/?", style="dim")
    header.append(f"  |  Elapsed: {_format_timestamp(elapsed)}", style="dim")

    if current_player:
        header.append(f"  |  Now: {current_player}", style="bold yellow")

    if paused:
        header.append("  |  [yellow]PAUSED (p = resume)[/]", style="yellow bold")

    if is_done:
        header.append("  |  [green]COMPLETE[/]", style="green bold")
        if winner:
            header.append(f"  |  Winner: {winner}", style="green")

    p1_panel = _render_chat_panel(
        f"Player 1: {player1_name}",
        player1_content,
        border_style="cyan",
    )
    p2_panel = _render_chat_panel(
        f"Player 2: {player2_name}",
        player2_content,
        border_style="magenta",
    )

    return Group(
        header,
        "",
        Group(p1_panel, p2_panel),
    )


def run_turn_chat(
    client: OllamaClient,
    player1_model: str,
    player2_model: str,
    initial_prompt: str,
    max_exchanges: int = DEFAULT_EXCHANGES,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    live: bool = True,
) -> tuple[list[dict], float, bool]:
    """
    Run turn-based chat between two models.

    Returns (conversation_log, total_elapsed, was_quit).
    conversation_log: list of {"player": str, "content": str, "elapsed": float}
    """
    log_path, log_str = _setup_log_file()
    start_time = time.perf_counter()
    _init_log_file(log_path, player1_model, player2_model, initial_prompt, start_time)

    conversation_log: list[dict] = []
    was_quit = False

    SYSTEM_PROMPT = "You are a helpful assistant."

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": initial_prompt}
    ]

    player1_content = initial_prompt
    player2_content = ""

    holder: dict = {
        "active": True,
        "paused": False,
        "quit": False,
        "skip_next": False,
        "submit_partial": False,
        "current_player": player1_model,
    }

    if live:
        key_thread = threading.Thread(target=_chat_key_reader, args=(holder,), daemon=True)
        key_thread.start()

    exchange_num = 0
    total_turns = max_exchanges * 2  # Each model gets max_exchanges turns

    while exchange_num < total_turns:
        exchange_num += 1

        if exchange_num % 2 == 1:
            current_model = player1_model
            current_player_name = player1_model
            other_player_name = player2_model
        else:
            current_model = player2_model
            current_player_name = player2_model
            other_player_name = player1_model

        holder["current_player"] = current_player_name
        holder["skipped"] = False
        holder["submit_partial"] = False

        console.print(f"[dim]Starting {current_player_name}...[/]")

        elapsed = time.perf_counter() - start_time

        if live:
            stream_holder: dict = {}

            def run_stream():
                _stream_chat_turn(client, current_model, messages, stream_holder)

            thread = threading.Thread(target=run_stream, daemon=True)
            thread.start()

            with Live(
                _build_status_panel(
                    exchange_num,
                    current_player_name,
                    player1_model,
                    player2_model,
                    player1_content,
                    player2_content,
                    elapsed,
                    holder.get("paused", False),
                    holder.get("paused_snapshot"),
                ),
                console=console,
                refresh_per_second=8,
            ) as live_ctx:
                while not stream_holder.get("done", False):
                    if holder.get("quit"):
                        was_quit = True
                        break

                    elapsed = time.perf_counter() - start_time
                    current_response = stream_holder.get("response", "")

                    if exchange_num % 2 == 1:
                        player1_content = current_response
                    else:
                        player2_content = current_response

                    live_ctx.update(
                        _build_status_panel(
                            exchange_num,
                            current_player_name,
                            player1_model,
                            player2_model,
                            player1_content,
                            player2_content,
                            elapsed,
                            holder.get("paused", False),
                            holder.get("paused_snapshot"),
                        )
                    )
                    time.sleep(0.125)

                holder["active"] = False
                thread.join(timeout=2.0)

                if holder.get("quit"):
                    was_quit = True

                if stream_holder.get("skipped"):
                    console.print(f"[yellow]Skipped by user[/]")
                    if exchange_num % 2 == 1:
                        player1_content = "[skipped]"
                    else:
                        player2_content = "[skipped]"
                elif stream_holder.get("error"):
                    console.print(f"[red]Error: {stream_holder['error']}[/]")
                    if exchange_num % 2 == 1:
                        player1_content = f"[error: {stream_holder['error']}]"
                    else:
                        player2_content = f"[error: {stream_holder['error']}]"

                final_response = stream_holder.get("response", "")
                elapsed = time.perf_counter() - start_time
                _append_to_log(log_path, exchange_num, current_player_name, final_response, elapsed)

                conversation_log.append({
                    "player": current_player_name,
                    "content": final_response,
                    "elapsed": elapsed,
                })

                messages.append({"role": "assistant", "content": final_response})
                messages.append({"role": "user", "content": final_response})

                if holder.get("quit"):
                    break
        else:
            elapsed = time.perf_counter() - start_time
            ok, err, response = _stream_chat_turn(client, current_model, messages, holder)

            if holder.get("quit"):
                was_quit = True
                break

            if ok:
                if exchange_num % 2 == 1:
                    player1_content = response
                else:
                    player2_content = response

                _append_to_log(log_path, exchange_num, current_player_name, response, elapsed)
                conversation_log.append({
                    "player": current_player_name,
                    "content": response,
                    "elapsed": elapsed,
                })
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": response})
            else:
                console.print(f"[red]Error: {err}[/]")
                if exchange_num % 2 == 1:
                    player1_content = f"[error: {err}]"
                else:
                    player2_content = f"[error: {err}]"

    total_elapsed = time.perf_counter() - start_time
    console.print(f"\n[green]Chat session complete. Total time: {_format_timestamp(total_elapsed)}[/]")
    console.print(f"[dim]Log saved to: {log_str}[/]")

    return conversation_log, total_elapsed, was_quit


def run_summary(
    client: OllamaClient,
    model: str,
    conversation_log: list[dict],
    prompt: str,
) -> tuple[bool, Optional[str], str]:
    """Run a summary query with the conversation as context."""
    context_parts = []
    for entry in conversation_log:
        role = "Player 1" if entry["player"] else entry["player"]
        context_parts.append(f"**{entry['player']}**: {entry['content']}")

    context = "\n\n".join(context_parts)
    full_prompt = f"""Here is a conversation between two AI models:

{context}

---

{prompt}"""

    messages = [{"role": "user", "content": full_prompt}]

    holder: dict = {}
    ok, err, response = _stream_chat_turn(client, model, messages, holder)

    if ok:
        return True, None, response
    return False, err, ""


def cmd_turn_chat(
    client: OllamaClient,
    player1: Optional[str] = None,
    player2: Optional[str] = None,
    prompt: Optional[str] = None,
    exchanges: int = DEFAULT_EXCHANGES,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> int:
    """Interactive turn-based chat between two models."""
    console.print()
    console.print(
        Panel(
            "[bold cyan]Ollama Model Manager[/] — [dim]Turn-Based Chat[/]",
            box=box.DOUBLE,
            border_style="cyan",
            padding=(0, 2),
        )
    )
    console.print()

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

    if not player1 or not player2:
        player1, player2 = _select_two_models(model_names)
        if not player1 or not player2:
            console.print("[yellow]Model selection cancelled.[/]")
            return 0
        console.print()
        console.print(
            Panel(
                f"Player 1: [cyan]{player1}[/]\nPlayer 2: [magenta]{player2}[/]",
                title="[cyan]Selected Models[/]",
                border_style="cyan",
                box=box.ROUNDED,
            )
        )
        console.print()

    if not prompt:
        prompt = _ask_initial_prompt()
        console.print()
        console.print(
            Panel(
                prompt,
                title="[cyan]Initial Prompt[/]",
                border_style="cyan",
                box=box.ROUNDED,
            )
        )
        console.print()

    env_config = collect_environment_config()
    _print_environment_panel(env_config)

    conversation_log, total_elapsed, was_quit = run_turn_chat(
        client,
        player1,
        player2,
        prompt,
        max_exchanges=exchanges,
        timeout_seconds=timeout,
    )

    console.print()
    summary_model = _ask_summary_model(model_names)
    if summary_model:
        summary_prompt = _ask_summary_prompt()
        console.print()
        console.print(f"[cyan]Running summary with {summary_model}...[/]")
        ok, err, summary = run_summary(client, summary_model, conversation_log, summary_prompt)
        if ok:
            console.print()
            console.print(
                Panel(
                    summary,
                    title=f"[cyan]Summary by {summary_model}[/]",
                    border_style="green",
                    box=box.ROUNDED,
                )
            )
        else:
            console.print(f"[red]Summary failed: {err}[/]")

    console.print()
    console.print("[dim]Session complete.[/]")
    return 0


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
