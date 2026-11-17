"""A deliberately small, read-mostly RunPod client.

Only what the runbook's commands need. Reads (balance, pods) are implemented here;
writes (create, terminate) are finished in B11 and exercised for the first time in
the M1 pilot. The API key is read from the secret store and sent in a header -- it
is never put on a command line, into a URL, or into a log line.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import requests

from ops.secrets import SecretStore

__all__ = ["RunPodClient", "RunPodError", "RunPodPod", "cheapest_on_demand_price",
           "image_exists", "price_for_cloud_type"]

# REST v2 since 11 Nov 2026 (v1 retires 2026-11-15). Pods and the GPU catalogue are on
# v2. The prepaid balance has no v2 equivalent and stays on GraphQL.
CLOUD_TYPES = ("COMMUNITY", "SECURE")

GRAPHQL_URL = "https://api.runpod.io/graphql"  # rp-migrate: keep-v1 (balance only)
REST_URL = "https://api.runpod.io/v2"
TIMEOUT = 30
# A cursor that never ends must not hang the "are any pods alive?" check for ever.
MAX_POD_PAGES = 50


def _uptime_seconds(started_at: Any, now: datetime | None = None) -> int:
    """Seconds since ``startedAt`` (ISO 8601, UTC). v2 dropped v1's ``uptimeSeconds``.

    A pod that has not started, or a timestamp we cannot read, is 0 seconds: the
    registry's cost estimate falls back to its own launch time in that case.
    """
    if not started_at:
        return 0
    try:
        started = datetime.fromisoformat(str(started_at).replace("Z", "+00:00"))
    except ValueError:
        return 0
    if started.tzinfo is None:
        started = started.replace(tzinfo=timezone.utc)
    return max(0, int(((now or datetime.now(timezone.utc)) - started).total_seconds()))


def _legacy_gpu_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """A v2 catalogue entry in the shape the price helpers below have always read.

    v2 lists a price for a tier even when that tier is not offered (A40: community
    price 0.35 with ``community: false``), so the tier flag decides, not the price.
    """
    price = entry.get("price") or {}
    return {
        "id": entry.get("id"),
        "displayName": entry.get("name"),  # was displayName
        "memoryInGb": entry.get("memory"),  # was memoryInGb
        "communityPrice": price.get("community") if entry.get("community") else None,
        "securePrice": price.get("secure") if entry.get("secure") else None,
        "availability": entry.get("availability"),
    }


class RunPodError(RuntimeError):
    """A RunPod API call that did not succeed."""


@dataclass(frozen=True)
class RunPodPod:
    """A pod as RunPod reports it."""

    pod_id: str
    name: str
    desired_status: str
    gpu_type: str
    gpu_count: int
    cost_per_hr: float
    uptime_seconds: int
    raw: dict[str, Any]

    @property
    def is_alive(self) -> bool:
        """Anything not terminated can cost money, EXITED pods included (storage)."""
        return self.desired_status.upper() not in ("TERMINATED",)


def cheapest_on_demand_price(entry: dict[str, Any]) -> float | None:
    """Cheapest usable on-demand price in a gpuTypes entry, or ``None``.

    RunPod reports an unavailable tier as price 0 rather than null, so a
    non-positive price means "not for sale", never "free". Spot prices are ignored:
    an interrupted pod costs wall-clock time this sprint does not have.
    """
    prices = []
    for field in ("communityPrice", "securePrice"):
        value = entry.get(field)
        if value is None:
            continue
        price = float(value)
        if price > 0:
            prices.append(price)
    return min(prices) if prices else None


DOCKER_HUB_TAG_URL = "https://hub.docker.com/v2/repositories/{repo}/tags/{tag}"


def image_exists(image: str, getter=None) -> bool | None:
    """Whether a Docker Hub tag exists. ``None`` means "could not check".

    RunPod accepts a create for an image that does not exist, fails the pull, and tears
    the pod down within seconds — so a typo in a tag looks exactly like capacity
    trouble and costs a launch window to diagnose. One HTTP call rules it out.

    ``None`` rather than ``False`` when the registry is unreachable or the image is not
    a tagged Docker Hub name: a third party being down must not block a launch, it must
    only stop us *claiming* the image was verified.
    """
    if "/" not in image or ":" not in image.rsplit("/", 1)[-1]:
        return None
    repo, _, tag = image.rpartition(":")
    if repo.count("/") != 1 or "." in repo.split("/")[0]:
        # A registry host (ghcr.io/...) or a bare library image: not Docker Hub's v2
        # repositories namespace, so we cannot check it this way.
        return None
    get = getter or requests.get
    try:
        response = get(DOCKER_HUB_TAG_URL.format(repo=repo, tag=tag), timeout=TIMEOUT)
    except Exception:
        return None
    if response.status_code == 200:
        return True
    if response.status_code == 404:
        return False
    return None


def price_for_cloud_type(entry: dict[str, Any], cloud_type: str) -> float | None:
    """On-demand price for one cloud tier, or ``None`` if that tier is unavailable.

    A job that asks for SECURE must be costed at the secure price. Quoting the cheapest
    tier and then being billed on the dearer one is how a run lands over its ceiling
    with every guard having passed.
    """
    if cloud_type not in CLOUD_TYPES:
        raise ValueError(f"cloud_type {cloud_type!r} is not one of {', '.join(CLOUD_TYPES)}")
    field = "communityPrice" if cloud_type == "COMMUNITY" else "securePrice"
    value = entry.get(field)
    if value is None:
        return None
    price = float(value)
    return price if price > 0 else None


class RunPodClient:
    def __init__(self, secrets: SecretStore, session: Any | None = None):
        self._key = secrets.require("RUNPOD_API_KEY")
        # Injectable so the create/terminate paths -- the two calls that cost and save
        # money -- are testable without an account. A test that needed live credentials
        # would not be run before a gate, which is exactly when it matters.
        self._session = session or requests

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._key}",
            "Content-Type": "application/json",
        }

    def _graphql(self, query: str) -> dict[str, Any]:
        response = self._session.post(
            GRAPHQL_URL, json={"query": query}, headers=self._headers, timeout=TIMEOUT
        )
        if response.status_code != 200:
            raise RunPodError(f"RunPod GraphQL returned HTTP {response.status_code}")
        payload = response.json()
        if payload.get("errors"):
            messages = "; ".join(e.get("message", "?") for e in payload["errors"])
            raise RunPodError(f"RunPod GraphQL error: {messages}")
        return payload["data"]

    def balance_usd(self) -> float:
        """Prepaid client balance in USD. This is the hard ceiling on all spend."""
        data = self._graphql("query { myself { clientBalance } }")  # rp-migrate: keep-v1
        return float(data["myself"]["clientBalance"])  # rp-migrate: keep-v1

    def pods(self) -> list[RunPodPod]:
        """Every pod the account lists, following the cursor to the last page."""
        items: list[dict[str, Any]] = []
        cursor: str | None = None
        for _ in range(MAX_POD_PAGES):
            response = self._session.get(
                f"{REST_URL}/pods", headers=self._headers, timeout=TIMEOUT,
                **({"params": {"cursor": cursor}} if cursor else {}),
            )
            if response.status_code != 200:
                raise RunPodError(
                    f"RunPod REST /pods returned HTTP {response.status_code}: "
                    f"{response.text[:200]}"
                )
            payload = response.json() or {}
            items.extend(payload.get("pods") or [])
            paging = payload.get("pagination") or {}
            cursor = paging.get("nextCursor")
            if not paging.get("hasNextPage") or not cursor:
                break
        else:
            # Reporting a partial list as complete could hide a live pod.
            raise RunPodError(
                f"RunPod /pods did not finish after {MAX_POD_PAGES} pages; refusing to "
                f"report a partial pod list"
            )
        return [
            RunPodPod(
                pod_id=str(item.get("id", "?")),
                name=str(item.get("name", "")),
                desired_status=str(item.get("status", "UNKNOWN")),  # was desiredStatus
                gpu_type=str((item.get("gpu") or {}).get("id") or "?"),
                gpu_count=int((item.get("gpu") or {}).get("count", 0) or 0),
                cost_per_hr=float(item.get("cost", 0) or 0),  # was costPerHr
                uptime_seconds=_uptime_seconds(item.get("startedAt")),
                raw=item,
            )
            for item in items
        ]

    def live_pods(self) -> list[RunPodPod]:
        return [p for p in self.pods() if p.is_alive]

    def gpu_types(self) -> list[dict[str, Any]]:
        """Catalogue of GPU types with community and secure on-demand prices."""
        response = self._session.get(
            f"{REST_URL}/catalog/gpus", headers=self._headers, timeout=TIMEOUT,
            params={"include": "AVAILABILITY", "product": "POD"},
        )
        if response.status_code != 200:
            raise RunPodError(
                f"RunPod REST /catalog/gpus returned HTTP {response.status_code}: "
                f"{response.text[:200]}"
            )
        return [_legacy_gpu_entry(e) for e in (response.json() or {}).get("gpus") or []]

    def gpu_price_usd_per_h(self, gpu_type: str, cloud_type: str | None = None) -> float | None:
        """Cheapest on-demand price for ``gpu_type``, or ``None`` if unavailable.

        Community Cloud is cheaper and is what the budget assumes; spot prices are
        deliberately ignored here because an interrupted pod costs us wall-clock
        time we do not have in a 30-day sprint.
        """
        wanted = gpu_type.strip().lower()
        for entry in self.gpu_types():
            names = {str(entry.get("id", "")).lower(), str(entry.get("displayName", "")).lower()}
            if wanted in names:
                if cloud_type is None:
                    return cheapest_on_demand_price(entry)
                return price_for_cloud_type(entry, cloud_type)
        return None

    # -- writes ----------------------------------------------------------------
    #
    # There is deliberately no stop/pause/suspend method. ``runpodctl stop pod``
    # leaves the pod's disk allocated and still billing, so every teardown path in
    # this project is a delete. ``ops/wrapper.py`` asserts the same property for the
    # on-pod script; ``tests/test_runpod_writes.py`` asserts it for this class, so
    # neither half can quietly acquire a stop path later.

    def create_pod(
        self,
        *,
        name: str,
        gpu_type: str,
        image: str,
        disk_gb: int,
        start_script: str,
        env: dict[str, str],
        gpu_count: int = 1,
        cloud_type: str = "COMMUNITY",
    ) -> str:
        """Create one on-demand pod and return its id.

        ``name`` is the run_id. A pod that escapes the registry is identified by its
        name and nothing else, so an unnamed pod is an unexplainable charge.
        """
        if not start_script.strip():
            raise ValueError(
                "refusing to create a pod with an empty start script: all three "
                "safety layers live in that script, so a pod without one runs until "
                "someone notices"
            )
        if cloud_type not in CLOUD_TYPES:
            raise ValueError(
                f"cloud_type {cloud_type!r} is not one of {', '.join(CLOUD_TYPES)}"
            )
        body = {
            "name": name,
            "image": image,  # was imageName
            "gpu": {"id": gpu_type, "count": gpu_count},  # was gpuTypeIds / gpuCount
            "disk": disk_gb,  # was containerDiskInGb
            "cloud": cloud_type,  # was cloudType; v2 defaults to SECURE, so always sent
            # Placement is left to whatever is free: no dataCenterIds. v1's
            # gpuTypePriority / dataCenterPriority are gone; we never sent a fallback
            # list (the hunt in ops/gpu.py tries one GPU type per create).
            #
            # No persistent volume: v1 needed volumeInGb 0 to say so, v2 says so by
            # leaving `mounts` out. Checkpoints go to the private Hugging Face repo.
            "cmd": ["bash", "-lc", start_script],  # was dockerStartCmd
            "env": dict(env),
        }
        response = self._session.post(
            f"{REST_URL}/pods", json=body, headers=self._headers, timeout=TIMEOUT
        )
        if response.status_code not in (200, 201):
            raise RunPodError(
                f"RunPod create pod returned HTTP {response.status_code}: "
                f"{response.text[:200]}"
            )
        payload = response.json() or {}
        pod_id = payload.get("id") or (payload.get("pod") or {}).get("id")
        if not pod_id:
            raise RunPodError(
                f"RunPod accepted the create but returned no pod id: "
                f"{str(payload)[:200]}. We may be billed for a pod whose id we do "
                f"not know — check the console now."
            )
        return str(pod_id)

    def terminate_pod(self, pod_id: str) -> bool:
        """Delete a pod. Returns True once it is gone.

        A 404 counts as success: teardown has to be idempotent, and treating "already
        gone" as a failure would turn a completed destroy into a retry loop.
        """
        response = self._session.delete(
            f"{REST_URL}/pods/{pod_id}", headers=self._headers, timeout=TIMEOUT
        )
        if response.status_code in (200, 202, 204, 404):  # v2 answers 204
            return True
        raise RunPodError(
            f"RunPod delete pod {pod_id} returned HTTP {response.status_code}: "
            f"{response.text[:200]}"
        )
