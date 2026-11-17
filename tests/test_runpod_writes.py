"""Tests for the write half of ops.runpod_client — creating and destroying pods.

These are the two calls that cost and save money. Both are tested against a fake
transport rather than the live API: a test that needed a real account would not be run
before a gate, which is precisely when it matters.
"""
import pytest

from ops.runpod_client import RunPodClient, RunPodError


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = text

    def json(self):
        return self._payload


class FakeSession:
    """Records every call so a test can assert what was sent, and to where."""

    def __init__(self, *responses):
        self._responses = list(responses)
        self.calls = []

    def _next(self):
        return self._responses.pop(0) if self._responses else FakeResponse()

    def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append(("POST", url, json))
        return self._next()

    def get(self, url, headers=None, timeout=None, params=None):
        self.calls.append(("GET", url, params))
        return self._next()

    def delete(self, url, headers=None, timeout=None):
        self.calls.append(("DELETE", url, None))
        return self._next()


class FakeSecrets:
    def require(self, name):
        return "rp-key-not-real"


def _client(*responses):
    session = FakeSession(*responses)
    return RunPodClient(FakeSecrets(), session=session), session


# --- delete, never stop -----------------------------------------------------------


def test_the_client_has_no_way_to_stop_a_pod():
    # runpodctl stop leaves the disk allocated and still billing. wrapper.py asserts
    # the same property for the on-pod script; this asserts it for the PC side, so
    # neither half can quietly acquire a stop path.
    forbidden = [name for name in dir(RunPodClient)
                 if any(word in name.lower() for word in ("stop", "pause", "suspend"))]
    assert forbidden == []


def test_terminate_issues_a_delete_to_the_pod_endpoint():
    client, session = _client(FakeResponse(200))
    assert client.terminate_pod("pod-123") is True
    method, url, _ = session.calls[0]
    assert method == "DELETE"
    assert url.endswith("/pods/pod-123")


def test_terminate_accepts_204_no_content():
    client, _ = _client(FakeResponse(204))
    assert client.terminate_pod("pod-123") is True


def test_terminate_treats_a_missing_pod_as_already_gone():
    # Teardown must be idempotent. A 404 means the thing we wanted gone is gone, and
    # raising here would make a successful destroy look like a failure and tempt a
    # retry loop.
    client, _ = _client(FakeResponse(404, text="not found"))
    assert client.terminate_pod("pod-123") is True


def test_terminate_raises_on_a_server_error_rather_than_claiming_success():
    # The worst outcome is reporting a pod destroyed when it is still billing.
    client, _ = _client(FakeResponse(500, text="boom"))
    with pytest.raises(RunPodError, match="500"):
        client.terminate_pod("pod-123")


def test_terminate_does_not_put_the_api_key_in_the_error_message():
    client, _ = _client(FakeResponse(500, text="boom"))
    with pytest.raises(RunPodError) as error:
        client.terminate_pod("pod-123")
    assert "rp-key-not-real" not in str(error.value)


# --- create -----------------------------------------------------------------------


def test_create_posts_the_spec_and_returns_the_pod_id():
    client, session = _client(FakeResponse(201, {"id": "pod-new"}))
    pod_id = client.create_pod(
        name="m1-pilot", gpu_type="NVIDIA GeForce RTX 4090", image="runpod/pytorch",
        disk_gb=60, start_script="#!/usr/bin/env bash\necho hi\n", env={"WANDB_PROJECT": "geetha"},
    )
    assert pod_id == "pod-new"
    method, url, body = session.calls[0]
    assert method == "POST"
    assert url.endswith("/pods")
    assert body["name"] == "m1-pilot"
    assert url.startswith("https://api.runpod.io/v2/")
    assert body["gpu"] == {"id": "NVIDIA GeForce RTX 4090", "count": 1}
    assert body["disk"] == 60
    assert body["image"] == "runpod/pytorch"
    assert body["cmd"] == ["bash", "-lc", "#!/usr/bin/env bash\necho hi\n"]


def test_create_names_the_pod_after_the_run_id():
    # The registry joins on pod id, but a pod that escaped the registry is identified
    # by its name. An unnamed pod is an unexplainable charge.
    client, session = _client(FakeResponse(201, {"id": "pod-new"}))
    client.create_pod(name="m2-stage-a", gpu_type="x", image="y", disk_gb=10,
                      start_script="#!/bin/bash\n", env={})
    assert session.calls[0][2]["name"] == "m2-stage-a"


def test_create_raises_when_the_response_carries_no_pod_id():
    # A create that half-succeeded is the one case where we may be billed for a pod
    # whose id we do not know, so it must be loud.
    client, _ = _client(FakeResponse(201, {"status": "ok"}))
    with pytest.raises(RunPodError, match="no pod id"):
        client.create_pod(name="r", gpu_type="x", image="y", disk_gb=10,
                          start_script="#!/bin/bash\n", env={})


def test_create_raises_on_a_rejected_request():
    client, _ = _client(FakeResponse(400, text="bad gpu type"))
    with pytest.raises(RunPodError, match="400"):
        client.create_pod(name="r", gpu_type="nope", image="y", disk_gb=10,
                          start_script="#!/bin/bash\n", env={})


def test_an_empty_start_script_is_refused_before_any_request_is_made():
    # A pod with no start script runs forever doing nothing: all three safety layers
    # live in that script.
    client, session = _client(FakeResponse(201, {"id": "pod-new"}))
    with pytest.raises(ValueError, match="start script"):
        client.create_pod(name="r", gpu_type="x", image="y", disk_gb=10,
                          start_script="   ", env={})
    assert session.calls == []


# --- placement constraints ------------------------------------------------------
#
# Live on 27 Oct, create returned HTTP 500 "There are no instances currently available"
# for both an RTX 4090 and an RTX A2000 — two very different tiers, which pointed at the
# request rather than at capacity. The OpenAPI schema showed why: cloudType defaults to
# SECURE and we were pinning COMMUNITY, and volumeInGb defaults to 20, so every request
# also demanded a persistent volume the probe has no use for.
#
# REST v2 (11 Nov): `cloud` still defaults to SECURE, so it is still always sent. "No
# volume" is now said by leaving `mounts` out; a zero size is rejected (10 GB floor).


def test_a_pod_asks_for_no_persistent_volume_by_default():
    # A throwaway pod that needs a 20 GB volume is a pod that can fail to place for a
    # reason unrelated to the GPU.
    client, session = _client(FakeResponse(201, {"id": "pod-new"}))
    client.create_pod(name="r", gpu_type="x", image="y", disk_gb=10,
                      start_script="#!/bin/bash\n", env={})
    assert "mounts" not in session.calls[0][2]


def test_the_cloud_type_is_explicit_and_defaults_to_the_cheaper_one():
    # Community is what the budget assumes; the caller can ask for secure when
    # availability matters more than price.
    client, session = _client(FakeResponse(201, {"id": "pod-new"}))
    client.create_pod(name="r", gpu_type="x", image="y", disk_gb=10,
                      start_script="#!/bin/bash\n", env={})
    assert session.calls[0][2]["cloud"] == "COMMUNITY"


def test_a_secure_pod_can_be_requested():
    client, session = _client(FakeResponse(201, {"id": "pod-new"}))
    client.create_pod(name="r", gpu_type="x", image="y", disk_gb=10,
                      start_script="#!/bin/bash\n", env={}, cloud_type="SECURE")
    assert session.calls[0][2]["cloud"] == "SECURE"


def test_an_unknown_cloud_type_is_refused_before_the_request():
    client, session = _client(FakeResponse(201, {"id": "pod-new"}))
    with pytest.raises(ValueError, match="cloud_type"):
        client.create_pod(name="r", gpu_type="x", image="y", disk_gb=10,
                          start_script="#!/bin/bash\n", env={}, cloud_type="CHEAP")
    assert session.calls == []


def test_placement_is_left_to_availability_rather_than_pinned():
    # No datacenter pinning: a probe that must land in one region is a probe that
    # cannot run when that region is full.
    client, session = _client(FakeResponse(201, {"id": "pod-new"}))
    client.create_pod(name="r", gpu_type="x", image="y", disk_gb=10,
                      start_script="#!/bin/bash\n", env={})
    body = session.calls[0][2]
    assert "dataCenterIds" not in body


def test_a_capacity_error_keeps_its_message_so_the_caller_can_recognise_it():
    client, _ = _client(FakeResponse(500, text='{"error":"create pod: There are no '
                                               'instances currently available"}'))
    with pytest.raises(RunPodError, match="no instances currently available"):
        client.create_pod(name="r", gpu_type="x", image="y", disk_gb=10,
                          start_script="#!/bin/bash\n", env={})


# --- the image has to exist -----------------------------------------------------
#
# Measured 27 Oct for INR 1.06: the spec carried
# "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04", which does not exist.
# RunPod accepted the create, failed the pull, and tore the pod down inside seven
# seconds. The same spec would have done the same thing at gate M1 — a wasted launch
# window rather than a wasted rupee. Checking the tag costs one HTTP call.


def test_a_tag_that_exists_is_accepted():
    from ops.runpod_client import image_exists

    calls = []

    def fake_get(url, timeout=None):
        calls.append(url)
        return FakeResponse(200, {"name": "1.4.1-cu1290-torch280-ubuntu2204"})

    assert image_exists("runpod/pytorch:1.4.1-cu1290-torch280-ubuntu2204",
                        getter=fake_get) is True
    assert "runpod/pytorch" in calls[0]


def test_a_tag_that_does_not_exist_is_rejected():
    from ops.runpod_client import image_exists

    assert image_exists("runpod/pytorch:nope", getter=lambda u, timeout=None:
                        FakeResponse(404, text="not found")) is False


def test_an_unreachable_registry_returns_unknown_rather_than_false():
    # A third party being down must not block a launch; it must only stop us claiming
    # the image was verified.
    from ops.runpod_client import image_exists

    def boom(url, timeout=None):
        raise OSError("dns failure")

    assert image_exists("runpod/pytorch:anything", getter=boom) is None


def test_an_image_with_no_tag_is_unknown_not_verified():
    # "runpod/pytorch" means :latest, which we never want to depend on anyway.
    from ops.runpod_client import image_exists

    assert image_exists("runpod/pytorch", getter=lambda *a, **k: FakeResponse(200)) is None


def test_a_non_dockerhub_image_is_unknown_rather_than_rejected():
    from ops.runpod_client import image_exists

    assert image_exists("ghcr.io/someone/thing:v1",
                        getter=lambda *a, **k: FakeResponse(200)) is None


# --- REST v2 shapes (migrated 11 Nov 2026; v1 retired 15 Nov) -------------------------


def test_create_sends_no_v1_field_name():
    # v2 rejects unknown fields with a 422, so a leftover v1 name is a failed launch.
    client, session = _client(FakeResponse(201, {"id": "pod-new"}))
    client.create_pod(name="r", gpu_type="x", image="y", disk_gb=10,
                      start_script="#!/bin/bash\n", env={})
    legacy = {"imageName", "gpuTypeIds", "gpuCount", "containerDiskInGb", "cloudType",
              "gpuTypePriority", "dataCenterPriority", "volumeInGb", "dockerStartCmd"}
    assert legacy.isdisjoint(session.calls[0][2])


def _pod(pod_id, status="RUNNING", **extra):
    return {"id": pod_id, "name": f"run-{pod_id}", "status": status, "cost": 0.59,
            "gpu": {"id": "NVIDIA A40", "count": 1}, "startedAt": None, **extra}


def test_pods_reads_the_v2_envelope_and_field_names():
    client, session = _client(FakeResponse(200, {
        "pods": [_pod("a")], "pagination": {"hasNextPage": False, "nextCursor": None}}))
    (pod,) = client.pods()
    assert session.calls[0][1] == "https://api.runpod.io/v2/pods"
    assert (pod.pod_id, pod.desired_status, pod.gpu_type, pod.gpu_count,
            pod.cost_per_hr) == ("a", "RUNNING", "NVIDIA A40", 1, 0.59)
    assert pod.is_alive


def test_pods_follows_the_cursor_so_a_live_pod_on_page_two_is_not_missed():
    # The "0 pods" check is a money guard. One page read as the whole list would
    # report nothing running while a pod on the next page bills.
    client, session = _client(
        FakeResponse(200, {"pods": [_pod("a", "TERMINATED")],
                           "pagination": {"hasNextPage": True, "nextCursor": "c1"}}),
        FakeResponse(200, {"pods": [_pod("b")],
                           "pagination": {"hasNextPage": False, "nextCursor": None}}))
    assert [p.pod_id for p in client.live_pods()] == ["b"]
    assert session.calls[1][2] == {"cursor": "c1"}


def test_pods_refuses_to_report_a_list_that_never_ends():
    from ops import runpod_client
    endless = [FakeResponse(200, {"pods": [], "pagination": {"hasNextPage": True,
                                                              "nextCursor": "again"}})
               for _ in range(runpod_client.MAX_POD_PAGES)]
    client, _ = _client(*endless)
    with pytest.raises(RunPodError, match="partial"):
        client.pods()


def test_provisioning_starting_and_error_pods_count_as_alive():
    # v2 added these states. Each can hold a disk and bill.
    client, _ = _client(FakeResponse(200, {
        "pods": [_pod("p", "PROVISIONING"), _pod("s", "STARTING"), _pod("e", "ERROR"),
                 _pod("x", "EXITED"), _pod("t", "TERMINATED")],
        "pagination": {"hasNextPage": False, "nextCursor": None}}))
    assert [p.pod_id for p in client.live_pods()] == ["p", "s", "e", "x"]


def test_uptime_comes_from_started_at():
    from datetime import datetime, timezone
    from ops.runpod_client import _uptime_seconds
    now = datetime(2026, 11, 11, 12, 0, 0, tzinfo=timezone.utc)
    assert _uptime_seconds("2026-11-11T11:00:00Z", now) == 3600
    assert _uptime_seconds("2026-11-11T11:59:30.250+00:00", now) == 29
    assert _uptime_seconds(None, now) == 0
    assert _uptime_seconds("not a time", now) == 0
    assert _uptime_seconds("2026-11-11T13:00:00Z", now) == 0   # clock skew, never negative


def test_the_catalogue_prices_only_the_tiers_that_are_offered():
    # Live on 11 Nov: the A40 lists a community price of 0.35 with `community: false`.
    # Quoting that price would cost a job on a tier it cannot be placed on.
    client, session = _client(FakeResponse(200, {"gpus": [
        {"id": "NVIDIA A40", "name": "A40", "memory": 48, "secure": True,
         "community": False, "price": {"community": 0.35, "secure": 0.59},
         "availability": "NONE"},
        {"id": "NVIDIA GeForce RTX 4090", "name": "RTX 4090", "memory": 24,
         "secure": True, "community": True,
         "price": {"community": 0.34, "secure": 0.89}, "availability": "LOW"}]}))
    assert client.gpu_price_usd_per_h("NVIDIA A40", "COMMUNITY") is None
    method, url, params = session.calls[0]
    assert url == "https://api.runpod.io/v2/catalog/gpus"
    assert params == {"include": "AVAILABILITY", "product": "POD"}
    client, _ = _client(FakeResponse(200, {"gpus": [
        {"id": "NVIDIA A40", "name": "A40", "memory": 48, "secure": True,
         "community": False, "price": {"community": 0.35, "secure": 0.59}}]}))
    assert client.gpu_price_usd_per_h("nvidia a40", "SECURE") == 0.59
    client, _ = _client(FakeResponse(200, {"gpus": [
        {"id": "NVIDIA GeForce RTX 4090", "name": "RTX 4090", "memory": 24,
         "secure": True, "community": True, "price": {"community": 0.34, "secure": 0.89}}]}))
    assert client.gpu_price_usd_per_h("RTX 4090") == 0.34
