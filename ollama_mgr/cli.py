"""CLI: check, list, benchmark, report, prune."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .api_client import OllamaClient
from .benchmark import ModelResult, run_all_benchmarks
from .report import format_report
from .prune import prune_models
from .interactive import cmd_run_interactive, DEFAULT_RUN_PROMPT
from .chat import cmd_turn_chat, DEFAULT_INITIAL_PROMPT, DEFAULT_EXCHANGES, DEFAULT_TIMEOUT_SECONDS


def cmd_check(client: OllamaClient) -> int:
    ok, err, info = client.health()
    if ok:
        print("Ollama is running.")
        if info:
            print("Version info:", info)
        return 0
    print("Ollama check failed:", err, file=sys.stderr)
    return 1


def cmd_list(client: OllamaClient) -> int:
    ok, err, models = client.list_models()
    if not ok:
        print("Failed to list models:", err, file=sys.stderr)
        return 1
    if not models:
        print("No models found.")
        return 0
    for m in models:
        name = m.get("name", "?")
        size = m.get("size")
        size_str = f"  ({size})" if size is not None else ""
        print(f"  {name}{size_str}")
    return 0


def cmd_benchmark(
    client: OllamaClient,
    report_path: str | None,
    test_prompt: str,
    nvidia_smi_interval: float,
) -> int:
    ok, err, models = client.list_models()
    if not ok:
        print("Failed to list models:", err, file=sys.stderr)
        return 1
    names = [m.get("name") for m in models if m.get("name")]
    if not names:
        print("No models to benchmark.")
        return 0
    print(f"Benchmarking {len(names)} model(s); nvidia-smi every {nvidia_smi_interval}s ...")
    results, samples, env_config = run_all_benchmarks(
        client, names, test_prompt=test_prompt, nvidia_smi_interval=nvidia_smi_interval
    )
    report_text = format_report(results, env_config, samples)
    print(report_text)
    if report_path:
        Path(report_path).write_text(report_text, encoding="utf-8")
        print(f"\nReport written to: {report_path}")
    return 0


def cmd_prune(client: OllamaClient, model_names: list[str]) -> int:
    if not model_names:
        print("No model names given. Use --models a,b,c or pass as positional args.", file=sys.stderr)
        return 1
    outcomes = prune_models(client, model_names)
    failed = False
    for name, success, err in outcomes:
        if success:
            print(f"Deleted: {name}")
        else:
            print(f"Failed to delete {name}: {err}", file=sys.stderr)
            failed = True
    return 1 if failed else 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Ollama model manager: check, list, benchmark, run, chat, prune")
    ap.add_argument("--base-url", default="http://localhost:11434", help="Ollama API base URL")
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="Check if Ollama is running")
    sub.add_parser("list", help="List all models")
    rp = sub.add_parser(
        "run",
        help="Interactive: select models (checkboxes), then run benchmark one at a time with verbose output",
    )
    rp.add_argument(
        "--prompt",
        default=None,
        metavar="PROMPT",
        help="Test prompt for each model; if omitted, you will be prompted for it when selecting models",
    )
    rp.add_argument(
        "--models", "-m",
        metavar="LIST",
        help="Comma-separated model names; if set, skip checkbox and run these (non-interactive)",
    )
    rp.add_argument(
        "--report", "-o",
        metavar="FILE",
        help="Write results to a markdown file (prompt + each model's stats and output)",
    )
    rp.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress live progress UI; print minimal progress lines only",
    )
    rp.add_argument(
        "--pause", "-p",
        type=float,
        default=15,
        metavar="SECONDS",
        help="Pause between models with countdown (default: 15); use 0 to disable",
    )
    bp = sub.add_parser("benchmark", help="Run all models, benchmark, and print report")
    bp.add_argument("--report", "-o", metavar="FILE", help="Also write report to file")
    bp.add_argument("--prompt", default="Say exactly: OK", help="Test prompt for benchmark")
    bp.add_argument("--nvidia-smi-interval", type=float, default=10.0, help="Run nvidia-smi every N seconds (default 10)")
    pr = sub.add_parser("prune", help="Delete (prune) models by name")
    pr.add_argument("model_names", nargs="*", metavar="MODEL", help="Model names to delete")
    pr.add_argument("--models", "-m", metavar="LIST", help="Comma-separated model names")
    cp = sub.add_parser(
        "chat",
        help="Turn-based chat: two models debate each other in a split-view UI",
    )
    cp.add_argument(
        "--player1", "-1",
        metavar="MODEL",
        help="Player 1 model name (left panel); if omitted, you will be prompted",
    )
    cp.add_argument(
        "--player2", "-2",
        metavar="MODEL",
        help="Player 2 model name (right panel); if omitted, you will be prompted",
    )
    cp.add_argument(
        "--prompt",
        default=None,
        metavar="PROMPT",
        help="Initial prompt to start the conversation; if omitted, you will be prompted",
    )
    cp.add_argument(
        "--exchanges", "-e",
        type=int,
        default=DEFAULT_EXCHANGES,
        metavar="N",
        help=f"Number of exchanges (turns per model, default: {DEFAULT_EXCHANGES})",
    )
    cp.add_argument(
        "--timeout", "-t",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        metavar="SECONDS",
        help=f"Timeout in seconds (default: {DEFAULT_TIMEOUT_SECONDS} = 1 hour)",
    )
    args = ap.parse_args()
    client = OllamaClient(base_url=args.base_url)
    if args.command == "check":
        return cmd_check(client)
    if args.command == "list":
        return cmd_list(client)
    if args.command == "run":
        model_list = None
        if getattr(args, "models", None):
            model_list = [x.strip() for x in args.models.split(",") if x.strip()]
        return cmd_run_interactive(
            client,
            prompt=getattr(args, "prompt", None),
            model_list=model_list,
            report_path=getattr(args, "report", None),
            quiet=getattr(args, "quiet", False),
            pause_seconds=getattr(args, "pause", 15),
        )
    if args.command == "benchmark":
        return cmd_benchmark(
            client,
            report_path=args.report,
            test_prompt=args.prompt,
            nvidia_smi_interval=args.nvidia_smi_interval,
        )
    if args.command == "prune":
        names = list(getattr(args, "model_names", []) or [])
        csv = getattr(args, "models", None)
        if csv:
            names.extend([x.strip() for x in csv.split(",") if x.strip()])
        if not names:
            print("No model names given. Use: prune MODEL [MODEL ...] or --models a,b,c", file=sys.stderr)
            return 1
        return cmd_prune(client, names)
    if args.command == "chat":
        return cmd_turn_chat(
            client,
            player1=getattr(args, "player1", None),
            player2=getattr(args, "player2", None),
            prompt=getattr(args, "prompt", None),
            exchanges=getattr(args, "exchanges", DEFAULT_EXCHANGES),
            timeout=getattr(args, "timeout", DEFAULT_TIMEOUT_SECONDS),
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
