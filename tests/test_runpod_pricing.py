"""Tests for the price picker in ops.runpod_client.

RunPod reports an unavailable tier as price 0, not as null. Reading that as "free"
would make a launch guard compare the ceiling against a cost of zero and wave
anything through, so 0 must mean unavailable.

Observed live on 2026-10-26: NVIDIA A100-SXM4-40GB came back with
communityPrice 1 and securePrice 0.
"""
from ops.runpod_client import cheapest_on_demand_price


def test_a_zero_price_tier_is_treated_as_unavailable_not_free():
    entry = {"id": "NVIDIA A100-SXM4-40GB", "communityPrice": 1, "securePrice": 0}
    assert cheapest_on_demand_price(entry) == 1.0


def test_the_cheaper_of_two_real_prices_wins():
    entry = {"id": "NVIDIA GeForce RTX 4090", "communityPrice": 0.34, "securePrice": 0.89}
    assert cheapest_on_demand_price(entry) == 0.34


def test_no_usable_price_returns_none():
    assert cheapest_on_demand_price({"id": "x", "communityPrice": 0, "securePrice": None}) is None


def test_a_missing_price_field_returns_none():
    assert cheapest_on_demand_price({"id": "x"}) is None


def test_spot_prices_are_ignored():
    # An interrupted pod costs wall-clock time we do not have in a 30-day sprint.
    entry = {"id": "x", "communityPrice": 0.50, "securePrice": None, "communitySpotPrice": 0.10}
    assert cheapest_on_demand_price(entry) == 0.50


# --- the price of the tier we will actually use ---------------------------------
#
# A spec that asks for SECURE capacity must be costed at the secure price. Quoting the
# cheapest tier and then billing on the dearer one is how a run lands over its ceiling
# with every guard having passed.

from ops.runpod_client import price_for_cloud_type  # noqa: E402

ENTRY = {"id": "NVIDIA L4", "communityPrice": 0.44, "securePrice": 0.59}


def test_the_community_price_is_used_for_a_community_job():
    assert price_for_cloud_type(ENTRY, "COMMUNITY") == 0.44


def test_the_secure_price_is_used_for_a_secure_job():
    assert price_for_cloud_type(ENTRY, "SECURE") == 0.59


def test_a_tier_priced_zero_is_unavailable_not_free():
    assert price_for_cloud_type({"communityPrice": 0, "securePrice": 0.5}, "COMMUNITY") is None


def test_a_missing_tier_price_is_none():
    assert price_for_cloud_type({"securePrice": 0.5}, "COMMUNITY") is None


def test_an_unknown_cloud_type_is_refused():
    import pytest

    with pytest.raises(ValueError):
        price_for_cloud_type(ENTRY, "CHEAP")
