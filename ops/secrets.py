r"""Loading E:\Geetha\.secrets\geetha.env without ever revealing it.

the project rules is absolute: these values are never printed, logged or committed. The
store therefore holds values behind explicit accessors and keeps them out of its
own ``repr`` -- the usual way a key escapes is a stack trace or a debug print of a
container that happened to hold it.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["DEFAULT_ENV_PATH", "SecretStore", "load_secrets"]

DEFAULT_ENV_PATH = Path(r"E:\Geetha\.secrets\geetha.env")

# Only these ever reach a pod (runbook section 9).
POD_ENV_ALLOWLIST = ("HF_TOKEN", "WANDB_API_KEY")


class SecretStore:
    """Name-addressable secrets whose values never appear in output."""

    __slots__ = ("_values", "_source")

    def __init__(self, values: dict[str, str], source: Path):
        self._values = {k: v for k, v in values.items() if v != ""}
        self._source = source

    def names(self) -> tuple[str, ...]:
        """The names present, sorted. Names are safe to show; values are not."""
        return tuple(sorted(self._values))

    def has(self, name: str) -> bool:
        return name in self._values

    def get(self, name: str, default: str | None = None) -> str | None:
        return self._values.get(name, default)

    def require(self, name: str) -> str:
        if name not in self._values:
            raise KeyError(
                f"{name} is missing or empty in {self._source}. Add it there; do not "
                "pass it on a command line."
            )
        return self._values[name]

    def export_to_env(self, names: tuple[str, ...]) -> None:
        """Put the named secrets into this process's environment."""
        for name in names:
            os.environ[name] = self.require(name)

    def pod_env(self) -> dict[str, str]:
        """The only secrets allowed onto a pod."""
        return {name: self.require(name) for name in POD_ENV_ALLOWLIST}

    def __repr__(self) -> str:
        return f"SecretStore(source={self._source}, names={list(self.names())})"

    __str__ = __repr__


def load_secrets(path: str | os.PathLike[str] | None = None) -> SecretStore:
    """Parse a ``KEY=VALUE`` env file. Comments, blanks and quotes are handled."""
    path = Path(path) if path is not None else DEFAULT_ENV_PATH
    if not path.exists():
        raise FileNotFoundError(f"secrets file not found at {path}")

    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        name = name.strip()
        if name.startswith("export "):
            name = name[len("export ") :].strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[name] = value
    return SecretStore(values, path)
