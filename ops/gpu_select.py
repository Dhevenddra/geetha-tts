r"""Choose a GPU at launch, from live stock and the gate's spend ceiling (D-045).

Runbook §2 prices every gate against an RTX 4090 at $0.34/h. On 3 Nov the live
catalogue had the 4090 **out of stock on both clouds**, with nothing under the ceiling
available at all — and twenty minutes later an RTX A6000 appeared at $0.33/h with
double the VRAM. A single ``gpu_type`` in a job spec is therefore a plan that fails on
the day, for reasons that have nothing to do with the job.

Two properties this gives the launcher:

**The ceiling is derived, not configured.** A gate approves a *spend*. INR 300 over 4 hours
is INR 75/h whatever the runbook assumed weeks earlier, so the affordability test is
computed from the approved amount and the planned hours every time.

**Order is preference, not price.** The first candidate that is both in stock and inside
the ceiling wins, even when a later one is cheaper — because 48 GB of VRAM decides
whether the pilot needs 4-bit quantisation, and that is a judgement the spec author
makes, not an arithmetic one. Listing A6000 before A4500 means "pay the extra for the
memory if it is there".

And it refuses rather than improvises: if nothing qualifies, there is no choice and
every rejection carries its reason, so the operator can be told *why* a launch did not
happen instead of discovering a dearer card on the invoice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

__all__ = ["GpuChoice", "Selection", "choose_gpu"]

# Anything the catalogue reports as one of these has no usable stock right now.
_NO_STOCK = {"unavailable", "out", "none", ""}


@dataclass(frozen=True)
class GpuChoice:
    gpu_id: str
    price_usd_per_hour: float
    price_inr_per_hour: float
    vram_gb: int | None
    stock: str


@dataclass(frozen=True)
class Selection:
    choice: GpuChoice | None
    rejections: tuple[str, ...]
    ceiling_inr_per_hour: float

    @property
    def summary(self) -> str:
        if self.choice is None:
            return (f"no GPU in stock under INR {self.ceiling_inr_per_hour:.2f}/h; "
                    f"{len(self.rejections)} candidate(s) rejected")
        return (f"{self.choice.gpu_id} at ${self.choice.price_usd_per_hour:.2f}/h "
                f"(INR {self.choice.price_inr_per_hour:.2f}/h, {self.choice.vram_gb} GB, "
                f"stock {self.choice.stock}) under a INR {self.ceiling_inr_per_hour:.2f}/h ceiling")


def choose_gpu(
    candidates: Sequence[str],
    *,
    catalogue: Iterable[Mapping[str, object]],
    max_inr: float,
    max_hours: float,
    usd_to_inr: float,
) -> Selection:
    """The first candidate that is in stock and affordable, or no choice at all."""
    if not len(candidates):
        raise ValueError("choose_gpu needs at least one candidate GPU")
    if max_hours <= 0:
        raise ValueError(f"max_hours must be positive, got {max_hours}")
    if max_inr <= 0:
        raise ValueError(f"max_inr must be positive, got {max_inr}")
    if usd_to_inr <= 0:
        raise ValueError(f"usd_to_inr must be positive, got {usd_to_inr}")

    ceiling = max_inr / max_hours
    by_id = {str(row.get("id")): row for row in catalogue}
    rejections: list[str] = []

    for name in candidates:
        row = by_id.get(name)
        if row is None:
            rejections.append(f"{name}: not in the catalogue at all")
            continue

        stock = str(row.get("stockStatus") or "").strip()
        price = row.get("pricePerHr")
        if stock.lower() in _NO_STOCK:
            rejections.append(f"{name}: no stock (status {stock or 'unknown'!r})")
            continue
        if price is None:
            # A null price with stock reported is not a bargain, it is missing data.
            rejections.append(f"{name}: stock {stock} but no price published")
            continue

        inr = float(price) * usd_to_inr
        if inr > ceiling:
            rejections.append(
                f"{name}: INR {inr:.2f}/h is over the INR {ceiling:.2f}/h this gate allows "
                f"({max_inr:.0f} over {max_hours:g} h)"
            )
            continue

        vram = row.get("memoryGb")
        return Selection(
            choice=GpuChoice(gpu_id=name, price_usd_per_hour=float(price),
                             price_inr_per_hour=inr,
                             vram_gb=int(vram) if vram is not None else None,
                             stock=stock),
            rejections=tuple(rejections),
            ceiling_inr_per_hour=ceiling,
        )

    return Selection(choice=None, rejections=tuple(rejections),
                     ceiling_inr_per_hour=ceiling)
