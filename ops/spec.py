"""Job specs: the readable description of what a paid run is allowed to cost.

One spec per run, at ``ops/jobs/<run_id>.yaml`` (runbook section 9). Validation is
deliberately strict and refuses anything it does not recognise: an unknown key is
usually a typo'd ceiling, and a ceiling that silently fails to apply is the exact
failure this project cannot afford.
"""

from __future__ import annotations

import re
from dataclasses import MISSING, dataclass, fields
from pathlib import Path
from typing import Any

__all__ = ["JobSpec", "PROVIDERS", "SpecError"]

PROVIDERS = ("runpod", "vast", "kaggle")

# Measured live 27 Oct 2026: an L4 create was refused on COMMUNITY and accepted on
# SECURE minutes apart, and the A2000, 3090 and 4090 were all unavailable on both at
# the time. Availability is per GPU type *and* per cloud type and moves hour to hour,
# so which one a job uses is a per-job decision with a price attached — secure is
# roughly 1.3-2.6x community — and therefore belongs in the spec.
CLOUD_TYPES = ("COMMUNITY", "SECURE")
GATES = ("M1", "M2", "M3", "M4", "M5", "M6")

# run_id names a directory under E:\Geetha\runs\ and is interpolated into pod
# commands, so it is restricted to characters that are safe in both places.
_RUN_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{1,62}$")


class SpecError(ValueError):
    """A job spec that must not be launched."""


def _as_gpu_candidates(raw: Any) -> tuple[str, ...]:
    """Parse ``gpu_candidates``. A bare string is a mistake, not a one-item list."""
    if isinstance(raw, str):
        raise SpecError(
            "gpu_candidates must be a list of GPU names, not a single string "
            f"({raw!r}); a bare string would silently become a list of characters"
        )
    try:
        names = [str(name) for name in raw]
    except TypeError as problem:
        raise SpecError(f"gpu_candidates must be a list of GPU names: {problem}") from problem
    for name in names:
        if not name.strip():
            raise SpecError("gpu_candidates contains an empty GPU name")
    return tuple(names)


@dataclass(frozen=True)
class JobSpec:
    run_id: str
    gate: str
    provider: str
    gpu_type: str
    image: str
    disk_gb: int
    max_hours: float
    max_cost_inr: float
    command: str
    data_repos: tuple[str, ...]
    checkpoint_repo: str
    wandb_project: str
    cloud_type: str = "COMMUNITY"
    # D-045: the GPU is chosen at launch from live stock and the gate's ceiling, in
    # this order of preference. `gpu_type` stays the last resort, so every spec already
    # on disk keeps working unchanged.
    gpu_candidates: tuple[str, ...] = ()

    @property
    def is_paid(self) -> bool:
        return self.provider != "kaggle"

    @property
    def candidate_gpus(self) -> tuple[str, ...]:
        """Preference order for the launcher: candidates first, ``gpu_type`` last.

        Order is preference, not price: VRAM decides whether a run needs 4-bit
        quantisation, and that is the spec author's judgement rather than arithmetic.
        """
        ordered: list[str] = []
        for name in (*self.gpu_candidates, self.gpu_type):
            if name not in ordered:
                ordered.append(name)
        return tuple(ordered)

    @property
    def ceiling_inr_per_hour(self) -> float:
        """What this run may pay per hour, derived from the gate rather than assumed."""
        return self.max_cost_inr / self.max_hours

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "JobSpec":
        expected = {f.name for f in fields(cls)}
        # A field with a default is optional: reporting it missing would break every
        # spec already on disk the moment an optional field is added.
        required = {
            f.name for f in fields(cls)
            if f.default is MISSING and f.default_factory is MISSING
        }
        missing = sorted(required - raw.keys())
        if missing:
            raise SpecError(f"job spec is missing required field(s): {', '.join(missing)}")
        unexpected = sorted(raw.keys() - expected)
        if unexpected:
            raise SpecError(f"job spec has unrecognised field(s): {', '.join(unexpected)}")

        spec = cls(
            run_id=str(raw["run_id"]),
            gate=str(raw["gate"]),
            provider=str(raw["provider"]),
            gpu_type=str(raw["gpu_type"]),
            image=str(raw["image"]),
            disk_gb=int(raw["disk_gb"]),
            max_hours=float(raw["max_hours"]),
            max_cost_inr=float(raw["max_cost_inr"]),
            command=str(raw["command"]),
            data_repos=tuple(raw["data_repos"]),
            checkpoint_repo=str(raw["checkpoint_repo"]),
            wandb_project=str(raw["wandb_project"]),
            cloud_type=str(raw.get("cloud_type", "COMMUNITY")),
            gpu_candidates=_as_gpu_candidates(raw.get("gpu_candidates", ())),
        )
        spec.validate()
        return spec

    @classmethod
    def from_yaml(cls, path: str | Path) -> "JobSpec":
        import yaml

        with open(path, encoding="utf-8") as handle:
            return cls.from_dict(yaml.safe_load(handle))

    def validate(self) -> None:
        if not _RUN_ID.match(self.run_id):
            raise SpecError(
                f"run_id {self.run_id!r} must be 2-63 characters of lowercase letters, "
                "digits, dot, dash or underscore: it names a directory and goes into "
                "pod commands"
            )
        if self.gate not in GATES:
            raise SpecError(f"gate {self.gate!r} is not one of {', '.join(GATES)}")
        if self.provider not in PROVIDERS:
            raise SpecError(f"provider {self.provider!r} is not one of {', '.join(PROVIDERS)}")
        if self.disk_gb <= 0:
            raise SpecError(f"disk_gb must be positive, got {self.disk_gb}")
        if self.max_hours <= 0:
            raise SpecError(f"max_hours must be positive, got {self.max_hours}")
        if self.provider == "kaggle":
            if self.max_cost_inr != 0:
                raise SpecError(
                    f"a kaggle job must have max_cost_inr 0 (it is the free tier), "
                    f"got {self.max_cost_inr}"
                )
        elif self.max_cost_inr <= 0:
            raise SpecError(
                f"a paid job needs a positive max_cost_inr ceiling, got {self.max_cost_inr}"
            )
        if self.cloud_type not in CLOUD_TYPES:
            raise SpecError(
                f"cloud_type {self.cloud_type!r} is not one of {', '.join(CLOUD_TYPES)}"
            )
        if not self.command.strip():
            raise SpecError("command must not be empty")
        if not self.checkpoint_repo.strip():
            raise SpecError("checkpoint_repo must be set: a run with nowhere to resume from is lost work")
