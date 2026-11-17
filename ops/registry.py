r"""What a launch writes down so that something else can kill the pod.

    E:\Geetha\manifests\pod_registry.csv

``pods_to_destroy`` decides on two numbers per pod: a deadline and a rupee ceiling.
RunPod knows neither -- they are our policy, not its API's -- so the launch records
them here, absolute and in one place, and every later command reads them back rather
than recomputing them.

Two decisions worth stating
---------------------------
**The deadline is stored absolute, not as "max_hours".** A duration would silently
restart from whenever something next read the file, which is exactly the shape of
failure that leaves a pod running overnight.

**A live pod with no record gets a ceiling of zero, not "unlimited".** The pod this
file cannot protect against is the one nobody wrote down -- a launch that bypassed the
guards, or a leftover from a crashed session. Returning "no limits recorded" for it
would make the unrecorded case the *safest* one to be in. So an unrecorded pod is
over its ceiling the moment it exists, and the watchdog destroys it on sight.

**Identity is the pod id and nothing else.** The first version of ``pod_states`` fell
back to matching on the pod's *name* when the id was unknown, so that an escaped pod
would at least have a readable label. A security review caught what that actually did:
a stray pod named ``m1-pilot`` inherited the real ``m1-pilot``'s rupee ceiling and
deadline, and therefore survived the watchdog. Names are chosen at create time and can
be reused by hand in the RunPod console, so that was an identity an accident could
forge. The name now appears only inside a human-readable label that is deliberately
shaped so it can never equal a burn-tracker run_id -- a label for the log line, never
an identity for an authorisation check. ``forget_pod`` keys on the pod id for the same
reason: keyed on run_id, forgetting one pod would drop a sibling row and hide a live
pod from the watchdog.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence

from ops.guards import PodState

__all__ = [
    "REGISTRY_COLUMNS",
    "REGISTRY_PATH",
    "RegistryError",
    "RegistryRow",
    "forget_pod",
    "pod_states",
    "read_registry",
    "record_launch",
    "unrecorded_label",
]

REGISTRY_PATH = Path(r"E:\Geetha\manifests\pod_registry.csv")

REGISTRY_COLUMNS = (
    "run_id",
    "pod_id",
    "gate",
    "provider",
    "launched_at",
    "deadline",
    "max_cost_inr",
    # The rate the launch was actually quoted. Pricing a finished run from the burn
    # row's *planned* rate under-recorded the spend whenever the price had moved, and
    # a missing rate priced the run at zero -- which quietly returns its share of the
    # gate's authorisation. The rate is known at launch, so it is written down there.
    "rate_usd_per_h",
)


def unrecorded_label(pod_id: str, name: str) -> str:
    """A readable identifier for a pod we have no record of.

    Shaped so it can never equal a burn-tracker run_id: ``pods_to_destroy`` destroys a
    pod whose run_id is unknown to the tracker, and a bare name would let a stray pod
    named after a real run pass that check.
    """
    return f"unrecorded:{pod_id}({name.strip() or 'unnamed'})"


class RegistryError(ValueError):
    """A pod registry that must not be used to make a teardown decision."""


@dataclass(frozen=True)
class RegistryRow:
    run_id: str
    pod_id: str
    gate: str
    provider: str
    launched_at: datetime
    deadline: datetime
    max_cost_inr: float
    rate_usd_per_h: float = 0.0


def _parse_time(value: str, where: str, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat((value or "").strip())
    except ValueError:
        raise RegistryError(f"{where}: {field} {value!r} is not an ISO timestamp") from None
    # Naive timestamps are read as UTC: a deadline compared against an aware "now"
    # would raise, and a watchdog that crashes is a watchdog that is not watching.
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _rate(value, where: str) -> float:
    text = (value or "").strip()
    if not text:
        return 0.0
    try:
        return float(text)
    except ValueError:
        raise RegistryError(f"{where}: rate_usd_per_h {value!r} is not a number") from None


def read_registry(path: str | Path = REGISTRY_PATH) -> list[RegistryRow]:
    """Every recorded live pod. A missing file reads as empty."""
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        header = tuple(reader.fieldnames or ())
        if header != REGISTRY_COLUMNS:
            raise RegistryError(
                f"pod registry column mismatch.\n"
                f"  expected: {', '.join(REGISTRY_COLUMNS)}\n"
                f"  found:    {', '.join(header)}"
            )
        rows: list[RegistryRow] = []
        for line_number, raw in enumerate(reader, start=2):
            run_id = (raw["run_id"] or "").strip()
            if not run_id:
                continue
            where = f"pod registry line {line_number}"
            try:
                ceiling = float((raw["max_cost_inr"] or "").strip())
            except ValueError:
                raise RegistryError(
                    f"{where}: max_cost_inr {raw['max_cost_inr']!r} is not a number"
                ) from None
            rows.append(
                RegistryRow(
                    run_id=run_id,
                    pod_id=(raw["pod_id"] or "").strip(),
                    gate=(raw["gate"] or "").strip(),
                    provider=(raw["provider"] or "").strip(),
                    launched_at=_parse_time(raw["launched_at"], where, "launched_at"),
                    deadline=_parse_time(raw["deadline"], where, "deadline"),
                    max_cost_inr=ceiling,
                    rate_usd_per_h=_rate(raw.get("rate_usd_per_h"), where),
                )
            )
    return rows


def record_launch(
    *,
    run_id: str,
    pod_id: str,
    gate: str,
    provider: str,
    max_hours: float,
    max_cost_inr: float,
    rate_usd_per_h: float = 0.0,
    launched_at: datetime | None = None,
    path: str | Path = REGISTRY_PATH,
) -> RegistryRow:
    """Write the deadline and ceiling for a pod that has just been created."""
    path = Path(path)
    existing = read_registry(path)
    if any(row.run_id == run_id for row in existing):
        raise RegistryError(
            f"{run_id!r} already has a live pod recorded. Destroy it (which forgets "
            f"the row) before launching again."
        )
    launched_at = launched_at or datetime.now(timezone.utc)
    row = RegistryRow(
        run_id=run_id,
        pod_id=pod_id,
        gate=gate,
        provider=provider,
        launched_at=launched_at,
        deadline=launched_at + timedelta(hours=max_hours),
        max_cost_inr=max_cost_inr,
        rate_usd_per_h=rate_usd_per_h,
    )
    fresh = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        if fresh:
            writer.writerow(REGISTRY_COLUMNS)
        writer.writerow([
            row.run_id, row.pod_id, row.gate, row.provider,
            row.launched_at.isoformat(), row.deadline.isoformat(),
            f"{row.max_cost_inr:.2f}", f"{row.rate_usd_per_h:.4f}",
        ])
    return row


def forget_pod(pod_id: str, *, path: str | Path = REGISTRY_PATH) -> None:
    """Drop one pod's row after it is destroyed, keyed on the **pod id**.

    Keyed on run_id instead, forgetting one pod would drop a sibling row and hide a
    live pod from the watchdog.

    Idempotent: teardown runs on paths where the row may already be gone, and raising
    there would strand a pod for the sake of tidiness.
    """
    path = Path(path)
    rows = [row for row in read_registry(path) if row.pod_id != pod_id]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(REGISTRY_COLUMNS)
        for row in rows:
            writer.writerow([
                row.run_id, row.pod_id, row.gate, row.provider,
                row.launched_at.isoformat(), row.deadline.isoformat(),
                f"{row.max_cost_inr:.2f}", f"{row.rate_usd_per_h:.4f}",
            ])


def pod_states(
    pods: Sequence,
    registry: Sequence[RegistryRow],
    *,
    inr_per_usd: float,
) -> list[PodState]:
    """Join the provider's live pods to the ceilings we recorded for them.

    Cost is computed from uptime and the pod's own hourly rate rather than read from
    the API, so a price that moved mid-run is reflected in the number the watchdog
    acts on.
    """
    by_pod = {row.pod_id: row for row in registry}
    states: list[PodState] = []
    for pod in pods:
        hours = pod.uptime_seconds / 3600.0
        cost = hours * pod.cost_per_hr * inr_per_usd
        # Only the pod id. A name match is not identity -- see the module docstring.
        row = by_pod.get(pod.pod_id)
        if row is None:
            states.append(
                PodState(
                    pod_id=pod.pod_id,
                    run_id=unrecorded_label(pod.pod_id, pod.name),
                    hours_used=hours,
                    cost_inr=cost,
                    max_cost_inr=0.0,
                    deadline=datetime.fromtimestamp(0, tz=timezone.utc),
                )
            )
        else:
            states.append(
                PodState(
                    pod_id=pod.pod_id,
                    run_id=row.run_id,
                    hours_used=hours,
                    cost_inr=cost,
                    max_cost_inr=row.max_cost_inr,
                    deadline=row.deadline,
                )
            )
    return states
