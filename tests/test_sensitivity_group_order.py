"""Tests for the sensitivity energy-group ordering convention."""

import numpy as np
import numpy.testing as npt
import pytest

from pyNDUS.sensitivity import Sensitivity


def _make_serpent_sensitivity(avg, rsd=None, group_structure=None):
    """Build a minimal Serpent Sensitivity object from ordered group data."""
    sens = Sensitivity.__new__(Sensitivity)
    sens.reader = "serpent"
    sens.energy_unit = "MeV"
    sens.responses = ["keff"]
    sens.materials = ["fuel"]
    sens.zaid = [922350]
    sens.zais = sens.zaid.keys()
    sens.MTs = ["xs 18"]
    if group_structure is None:
        group_structure = [1.0, 2.0, 3.0, 4.0]

    sens.group_structure = np.asarray(group_structure, dtype=float)

    avg = np.asarray(avg, dtype=float)
    rsd = np.zeros_like(avg) if rsd is None else np.asarray(rsd, dtype=float)
    raw = np.zeros((1, 1, 1, avg.size, 2), dtype=float)
    raw[0, 0, 0, :, 0] = avg
    raw[0, 0, 0, :, 1] = rsd

    sens.sens = {"keff": raw}
    sens.sens_rsd = {"keff": raw}
    return sens


def _make_eranos_sensitivity(raw_descending):
    """Build a minimal ERANOS Sensitivity object from high-to-low group data."""
    sens = Sensitivity.__new__(Sensitivity)
    sens.reader = "eranos"
    sens.energy_unit = "eV"
    sens.responses = ["keff"]
    sens.materials = ["REACTOR"]
    sens.zaid = [922350]
    sens.zais = sens.zaid.keys()
    sens.MTs = [18]
    sens.group_structure = np.array([1.0, 2.0, 3.0, 4.0])
    sens.sens_rsd = None
    sens.sens = {
        "keff": {
            "REACTOR": {
                "U-235": np.asarray([raw_descending], dtype=float)
            }
        }
    }
    return sens


def test_serpent_profiles_are_stored_in_ascending_energy_order():
    """Preserve Serpent profiles that already follow the ascending energy grid."""
    sens = _make_serpent_sensitivity(avg=[1.0, 2.0, 3.0], rsd=[0.1, 0.2, 0.3])

    npt.assert_allclose(sens.sens.reshape(-1), [1.0, 2.0, 3.0])
    npt.assert_allclose(sens.sens_rsd.reshape(-1), [0.1, 0.2, 0.3])


def test_eranos_profiles_are_reordered_to_ascending_energy_order():
    """Reverse ERANOS high-to-low tables before storing them internally."""
    sens = _make_eranos_sensitivity(raw_descending=[3.0, 2.0, 1.0])

    npt.assert_allclose(sens.sens.reshape(-1), [1.0, 2.0, 3.0])


def test_get_returns_ascending_order_by_default_and_descending_on_request():
    """Expose ascending profiles by default while keeping a descending view."""
    sens = _make_serpent_sensitivity(avg=[1.0, 2.0, 3.0], rsd=[0.1, 0.2, 0.3])

    avg, rsd = sens.get(resp=["keff"], mat=["fuel"], za=[922350], MT=[18])
    avg_desc, rsd_desc = sens.get(resp=["keff"], mat=["fuel"], za=[922350], MT=[18],
                                  group_order="descending",
                                  )

    npt.assert_allclose(avg.reshape(-1), [1.0, 2.0, 3.0])
    npt.assert_allclose(rsd.reshape(-1), [0.1, 0.2, 0.3])
    npt.assert_allclose(avg_desc.reshape(-1), [3.0, 2.0, 1.0])
    npt.assert_allclose(rsd_desc.reshape(-1), [0.3, 0.2, 0.1])


def test_get_group_numbers_follow_requested_order():
    """Interpret group numbers in the same order requested for the output."""
    sens = _make_serpent_sensitivity(avg=[1.0, 2.0, 3.0], rsd=[0.1, 0.2, 0.3])

    avg_first_asc, _ = sens.get(resp=["keff"], mat=["fuel"], za=[922350], MT=[18], g=1)
    avg_first_desc, _ = sens.get(resp=["keff"], mat=["fuel"], za=[922350], MT=[18], g=1,
                                 group_order="descending",
                                 )
    avg_list_desc, _ = sens.get(resp=["keff"], mat=["fuel"], za=[922350], MT=[18], g=[1, 3],
                                group_order="descending",
                                )

    npt.assert_allclose(avg_first_asc.reshape(-1), [1.0])
    npt.assert_allclose(avg_first_desc.reshape(-1), [3.0])
    npt.assert_allclose(avg_list_desc.reshape(-1), [3.0, 1.0])


def test_get_rejects_unknown_group_order():
    """Reject ambiguous ordering requests explicitly."""
    sens = _make_serpent_sensitivity(avg=[1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="group_order"):
        sens.get(resp=["keff"], mat=["fuel"], za=[922350], MT=[18], group_order="upwards")


def test_group_structure_is_normalized_to_ascending_order():
    """Store descending boundary vectors in ascending order."""
    sens = _make_serpent_sensitivity(avg=[1.0, 2.0, 3.0],
                                     group_structure=[4.0, 3.0, 2.0, 1.0],
                                     )

    npt.assert_allclose(sens.group_structure, [1.0, 2.0, 3.0, 4.0])


def test_normalize_sens_profile_accepts_ascending_and_descending_grids():
    """Normalize with positive lethargy widths regardless of grid direction."""
    ascending = [1.0, np.e, np.e**2]
    descending = ascending[::-1]

    npt.assert_allclose(Sensitivity.NormalizeSensProfile([2.0, 4.0], ascending), [2.0, 4.0])
    npt.assert_allclose(Sensitivity.NormalizeSensProfile([2.0, 4.0], descending), [2.0, 4.0])


def test_normalize_sens_profile_rejects_inconsistent_grid():
    """Raise when the sensitivity length does not match the energy grid."""
    with pytest.raises(ValueError, match="Energy grid"):
        Sensitivity.NormalizeSensProfile([1.0, 2.0], [1.0, 2.0])
