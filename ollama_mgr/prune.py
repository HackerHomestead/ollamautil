"""Delete (prune) models by name."""

from __future__ import annotations

from .api_client import OllamaClient


def prune_models(client: OllamaClient, model_names: list[str]) -> list[tuple[str, bool, str | None]]:
    """
    Delete each model by name. Returns list of (name, success, error_message).
    """
    out = []
    for name in model_names:
        ok, err = client.delete_model(name)
        out.append((name, ok, err))
    return out
