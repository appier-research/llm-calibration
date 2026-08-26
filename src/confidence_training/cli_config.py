"""JSON-to-argv expansion for small one-run utility entry points."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Sequence


def inject_json_config(argv: Sequence[str] | None = None) -> None:
    """Inject JSON fields absent from argv, preserving explicit CLI precedence.

    This is intentionally used only by utility CLIs whose JSON keys are direct
    snake-case equivalents of their long options. Training and construction have
    typed resolvers because their configuration contracts are richer.
    """

    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        index = arguments.index("--config")
    except ValueError:
        return
    if index + 1 >= len(arguments):
        raise ValueError("--config requires a JSON path")
    loaded = json.loads(Path(arguments[index + 1]).read_text())
    if not isinstance(loaded, dict):
        raise ValueError("CLI configuration must be a JSON object")

    existing = {token for token in arguments if token.startswith("--")}
    injected: list[str] = []
    for key, value in loaded.items():
        option = "--" + key.replace("_", "-")
        if option in existing or value is None:
            continue
        if isinstance(value, bool):
            if value:
                injected.append(option)
            continue
        if isinstance(value, list):
            if key in {"reference", "runs_root", "final_runs_root"}:
                for item in value:
                    injected.extend((option, str(item)))
            else:
                injected.append(option)
                injected.extend(str(item) for item in value)
            continue
        if isinstance(value, (dict, tuple)):
            injected.extend((option, json.dumps(value)))
            continue
        injected.extend((option, str(value)))
    sys.argv.extend(injected)

