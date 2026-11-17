"""Choosing a GPU at launch from live stock and the gate's ceiling (D-045).

The runbook prices every gate against an RTX 4090 at $0.34/h. On 3 Nov the live
catalogue had the 4090 out of stock on both clouds, with nothing under the ceiling
available at all — and twenty minutes later an A6000 appeared at $0.33/h with double
the VRAM. So the GPU cannot be a constant in the job spec.

The ceiling is derived from the gate, not configured: a gate approves a *spend*, and
₹300 over 4 hours is ₹75/h whatever the runbook assumed months earlier.
"""

from __future__ import annotations

import pytest

from ops.gpu_select import choose_gpu

USD_TO_INR = 88.0


def catalogue(**overrides) -> list[dict]:
    rows = {
        "NVIDIA GeForce RTX 4090": dict(stockStatus="unavailable", pricePerHr=None, memoryGb=24),
        "NVIDIA RTX A6000": dict(stockStatus="Low", pricePerHr=0.33, memoryGb=48),
        "NVIDIA RTX A4500": dict(stockStatus="Low", pricePerHr=0.19, memoryGb=20),
        "NVIDIA L40S": dict(stockStatus="Low", pricePerHr=1.09, memoryGb=48),
    }
    rows.update(overrides)
    return [{"id": name, **fields} for name, fields in rows.items()]


def select(candidates, **kwargs):
    spec = dict(catalogue=catalogue(), max_inr=300.0, max_hours=4.0, usd_to_inr=USD_TO_INR)
    spec.update(kwargs)
    return choose_gpu(candidates, **spec)


class TestChoosing:
    def test_the_first_candidate_that_is_in_stock_and_affordable_wins(self):
        # Order is preference, not price: A6000's 48 GB is worth more to this job than
        # A4500's lower price, and the spec says so by listing it first.
        result = select(["NVIDIA GeForce RTX 4090", "NVIDIA RTX A6000", "NVIDIA RTX A4500"])
        assert result.choice is not None
        assert result.choice.gpu_id == "NVIDIA RTX A6000"
        assert result.choice.price_usd_per_hour == pytest.approx(0.33)

    def test_an_out_of_stock_candidate_is_skipped_and_the_reason_recorded(self):
        result = select(["NVIDIA GeForce RTX 4090", "NVIDIA RTX A6000"])
        assert any("4090" in reason and "stock" in reason.lower()
                   for reason in result.rejections)

    def test_a_candidate_over_the_hourly_ceiling_is_skipped(self):
        result = select(["NVIDIA L40S", "NVIDIA RTX A6000"])
        assert result.choice.gpu_id == "NVIDIA RTX A6000"
        assert any("L40S" in reason for reason in result.rejections)

    def test_the_inr_price_is_reported_so_the_burn_tracker_can_use_it(self):
        result = select(["NVIDIA RTX A6000"])
        assert result.choice.price_inr_per_hour == pytest.approx(0.33 * USD_TO_INR)

    def test_the_vram_is_carried_through_because_it_decides_quantisation(self):
        assert select(["NVIDIA RTX A6000"]).choice.vram_gb == 48


class TestNothingQualifies:
    def test_no_affordable_stock_returns_no_choice_rather_than_the_cheapest_anyway(self):
        result = select(["NVIDIA GeForce RTX 4090", "NVIDIA L40S"])
        assert result.choice is None

    def test_every_rejection_is_explained(self):
        result = select(["NVIDIA GeForce RTX 4090", "NVIDIA L40S"])
        assert len(result.rejections) == 2

    def test_a_candidate_absent_from_the_catalogue_is_reported_by_name(self):
        result = select(["NVIDIA MadeUp 9000"])
        assert result.choice is None
        assert any("MadeUp" in reason for reason in result.rejections)

    def test_an_empty_candidate_list_is_refused(self):
        with pytest.raises(ValueError, match="candidate"):
            select([])


class TestCeilingArithmetic:
    def test_the_ceiling_comes_from_the_gate_not_from_a_constant(self):
        # ₹300 over 4 h is ₹75/h ≈ $0.85/h, so $0.33 fits and $1.09 does not.
        tight = select(["NVIDIA L40S"], max_inr=300.0, max_hours=4.0)
        assert tight.choice is None
        loose = select(["NVIDIA L40S"], max_inr=600.0, max_hours=4.0)
        assert loose.choice is not None

    def test_fewer_hours_buys_a_dearer_card_within_the_same_gate(self):
        # The real trade: 2 h on an L40S fits ₹300 where 4 h does not.
        assert select(["NVIDIA L40S"], max_hours=2.0).choice is not None

    def test_a_null_price_is_treated_as_unavailable_not_as_free(self):
        rows = catalogue()
        rows.append({"id": "NVIDIA Mystery", "stockStatus": "High", "pricePerHr": None,
                     "memoryGb": 24})
        result = choose_gpu(["NVIDIA Mystery"], catalogue=rows, max_inr=300.0,
                            max_hours=4.0, usd_to_inr=USD_TO_INR)
        assert result.choice is None

    def test_non_positive_hours_is_refused(self):
        with pytest.raises(ValueError, match="max_hours"):
            select(["NVIDIA RTX A6000"], max_hours=0.0)

    def test_the_ceiling_is_reported_so_a_refusal_can_be_explained_to_the_operator(self):
        result = select(["NVIDIA L40S"])
        assert result.ceiling_inr_per_hour == pytest.approx(75.0)
