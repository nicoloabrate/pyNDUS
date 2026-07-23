from collections import OrderedDict

import numpy as np
import pytest

from pyNDUS import Sensitivity, SensitivityAlgebraError


def sensitivity(
    values,
    rsd=0.1,
    *,
    responses=("keff",),
    materials=("fuel",),
    zaids=(922350,),
    mts=(18,),
    groups=(1.0e-11, 1.0, 20.0),
):
    result = Sensitivity.__new__(Sensitivity)
    result._responses = tuple(responses)
    result._materials = OrderedDict((value, i) for i, value in enumerate(materials))
    result._zaid = OrderedDict((value, i) for i, value in enumerate(zaids))
    result._zais = OrderedDict((f"ZA-{value}", i) for i, value in enumerate(zaids))
    result._MTs = OrderedDict((value, i) for i, value in enumerate(mts))
    result._group_structure = np.asarray(groups)
    result._sens = np.asarray(values, dtype=float).reshape(
        len(responses), len(materials), len(zaids), len(mts), len(groups) - 1
    )
    if rsd is None:
        result._sens_rsd = None
    else:
        result._sens_rsd = np.broadcast_to(rsd, result.sens.shape).copy()
    return result


def test_halves_reconstruct_same_average_and_uncertainty():
    original = sensitivity([2.0, -4.0], rsd=0.25)

    reconstructed = original / 2 + original / 2

    np.testing.assert_allclose(reconstructed.sens, original.sens)
    np.testing.assert_allclose(reconstructed.sens_rsd, original.sens_rsd)


def test_power_scales_sensitivity_and_zero_power_contributes_nothing():
    first = sensitivity([2.0, -4.0], rsd=0.25)
    second = sensitivity([7.0, 9.0], rsd=0.5)

    reconstructed = first**1 + second**0

    np.testing.assert_allclose(reconstructed.sens, first.sens)
    np.testing.assert_allclose(reconstructed.sens_rsd, first.sens_rsd)


def test_distinct_sources_are_combined_as_independent():
    first = sensitivity([2.0, 2.0], rsd=0.1)
    second = sensitivity([3.0, 3.0], rsd=0.2)

    total = first + second

    expected_absolute_std = np.sqrt((2.0 * 0.1) ** 2 + (3.0 * 0.2) ** 2)
    np.testing.assert_allclose(total.sens, 5.0)
    np.testing.assert_allclose(total.sens_rsd, expected_absolute_std / 5.0)


def test_raise_policy_accepts_reordered_equal_metadata_and_aligns_values():
    left = sensitivity(
        [1.0, 2.0, 10.0, 20.0],
        responses=("keff", "beff"),
    )
    right = sensitivity(
        [100.0, 200.0, 1000.0, 2000.0],
        responses=("beff", "keff"),
    )

    total = left + right

    np.testing.assert_allclose(total.sens[:, 0, 0, 0, :], [[1001, 2002], [110, 220]])
    assert total.responses == left.responses


def test_raise_policy_reports_different_metadata():
    left = sensitivity([1.0, 2.0], zaids=(922350,))
    right = sensitivity([3.0, 4.0], zaids=(922380,))

    with pytest.raises(SensitivityAlgebraError, match="zaid"):
        left + right


def test_intersect_keeps_common_metadata_in_left_order_without_mutating_inputs():
    left = sensitivity(
        np.arange(8),
        responses=("keff", "beff"),
        zaids=(922350, 922380),
    )
    right = sensitivity(
        np.arange(8) + 10,
        responses=("beff", "leff"),
        zaids=(922380, 942390),
    )

    total = left.combine(right, policy="intersect")

    assert total.responses == ("beff",)
    assert tuple(total.zaid) == (922380,)
    assert total.sens.shape == (1, 1, 1, 1, 2)
    np.testing.assert_allclose(total.sens.ravel(), [16, 18])
    assert left.responses == ("keff", "beff")
    assert tuple(right.zaid) == (922380, 942390)


def test_with_policy_enables_intersection_and_preserves_uncertainty_provenance():
    original = sensitivity(
        np.arange(8) + 1,
        responses=("keff", "beff"),
        zaids=(922350, 922380),
    )
    narrowed = sensitivity(
        [100.0, 200.0],
        responses=("keff",),
        zaids=(922350,),
    )

    total = original.with_algebra_policy("intersect") / 2 + narrowed**0

    np.testing.assert_allclose(total.sens.ravel(), [0.5, 1.0])
    np.testing.assert_allclose(total.sens_rsd, 0.1)


def test_group_structures_must_always_match():
    left = sensitivity([1.0, 2.0])
    right = sensitivity([3.0, 4.0], groups=(1.0e-11, 2.0, 20.0))

    with pytest.raises(SensitivityAlgebraError, match="group structures"):
        left.combine(right, policy="intersect")


def test_zero_policy_uses_union_and_zero_fills_every_missing_axis():
    left = sensitivity(
        [2.0, 4.0],
        rsd=0.1,
        responses=("keff",),
        materials=("fuel",),
        zaids=(922350,),
        mts=(18,),
    )
    right = sensitivity(
        [3.0, 6.0],
        rsd=0.2,
        responses=("beff",),
        materials=("coolant",),
        zaids=(942390,),
        mts=(102,),
    )

    total = left.combine(right, policy="zero")

    assert total.responses == ("keff", "beff")
    assert tuple(total.materials) == ("fuel", "coolant")
    assert tuple(total.zaid) == (922350, 942390)
    assert tuple(total.zais) == ("ZA-922350", "ZA-942390")
    assert tuple(total.MTs) == (18, 102)
    assert total.sens.shape == (2, 2, 2, 2, 2)
    np.testing.assert_allclose(total.sens[0, 0, 0, 0], [2.0, 4.0])
    np.testing.assert_allclose(total.sens[1, 1, 1, 1], [3.0, 6.0])

    missing = total.sens.copy()
    missing[0, 0, 0, 0] = 0.0
    missing[1, 1, 1, 1] = 0.0
    np.testing.assert_allclose(missing, 0.0)
    np.testing.assert_allclose(total.sens_rsd[0, 0, 0, 0], 0.1)
    np.testing.assert_allclose(total.sens_rsd[1, 1, 1, 1], 0.2)
    zero_rsd = total.sens_rsd.copy()
    zero_rsd[0, 0, 0, 0] = 0.0
    zero_rsd[1, 1, 1, 1] = 0.0
    np.testing.assert_allclose(zero_rsd, 0.0)


def test_zero_policy_preserves_profile_missing_from_other_sensitivity():
    first = sensitivity([2.0, -4.0], rsd=0.25, zaids=(922350,))
    second = sensitivity([7.0, 9.0], rsd=0.5, zaids=(942390,))

    total = first.with_algebra_policy("zero") + second

    np.testing.assert_allclose(total.get(za=922350)[0], first.sens)
    np.testing.assert_allclose(total.get(za=922350)[1], first.sens_rsd)
    np.testing.assert_allclose(total.get(za=942390)[0], second.sens)
    np.testing.assert_allclose(total.get(za=942390)[1], second.sens_rsd)


def test_response_product_and_division_follow_log_sensitivity_rules():
    first = sensitivity([2.0, 4.0], rsd=None)
    second = sensitivity([0.5, 1.0], rsd=None)

    np.testing.assert_allclose((first * second).sens, first.sens + second.sens)
    np.testing.assert_allclose((first / second).sens, first.sens - second.sens)


def test_in_place_data_change_starts_new_uncertainty_provenance():
    original = sensitivity([2.0, 4.0], rsd=0.1)
    derived = original / 2
    derived._sens *= 2

    total = original + derived

    expected_absolute_std = np.sqrt((original.sens * 0.1) ** 2 + (derived.sens * 0.1) ** 2)
    np.testing.assert_allclose(
        total.sens_rsd,
        expected_absolute_std / np.abs(total.sens),
    )
