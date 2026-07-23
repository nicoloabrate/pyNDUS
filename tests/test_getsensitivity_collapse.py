"""Tests for energy collapsing in Sensitivity objects."""

import numpy as np
import numpy.testing as npt
import pytest

from pyNDUS.sensitivity import Sensitivity, SensitivityError
from pyNDUS import utils


def _make_sensitivity(avg, rsd=None):
    """Build a minimal Sensitivity object with one response/material/ZA/MT."""
    sens = Sensitivity.__new__(Sensitivity)
    sens.reader = "serpent"
    sens.energy_unit = "MeV"
    sens.responses = ["keff"]
    sens.materials = ["fuel"]
    sens.zaid = [922350]
    sens.zais = sens.zaid.keys()
    sens.MTs = [18]
    sens.group_structure = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    sens._sens = np.asarray(avg, dtype=float).reshape(1, 1, 1, 1, -1)

    if rsd is None:
        sens._sens_rsd = None
    else:
        sens._sens_rsd = np.asarray(rsd, dtype=float).reshape(1, 1, 1, 1, -1)
    return sens


def test_collapse_weighted_deterministic_profile():
    """Collapse deterministic sensitivities with the provided group weights."""
    sens = _make_sensitivity([1.0, 2.0, 3.0, 4.0])
    original_grid = sens.group_structure.copy()

    sens.collapse(fewgrp=[1.0, 3.0, 5.0], weight=np.array([1.0, 2.0, 3.0, 4.0]),
                  egridname="two-group",
                  )

    npt.assert_allclose(sens.sens.reshape(-1), [5.0, 25.0])
    assert sens.sens_rsd is None
    npt.assert_allclose(sens.fine_energygrid, original_grid)
    assert sens.fine_energygrid_unit == "MeV"
    npt.assert_allclose(sens.group_structure, [1.0, 3.0, 5.0])
    assert sens.egridname == "two-group"


def test_collapse_propagates_relative_standard_deviation():
    """Collapse stochastic sensitivities and propagate RSD in quadrature."""
    sens = _make_sensitivity(avg=[1.0, 2.0, 3.0, 4.0], rsd=[0.1, 0.1, 0.2, 0.2], )

    sens.collapse(fewgrp=[1.0, 3.0, 5.0])

    expected_rsd = [np.sqrt(0.2**2 + 0.4**2) / 3.0, np.sqrt(1.2**2 + 1.6**2) / 7.0, ]

    npt.assert_allclose(sens.sens.reshape(-1), [3.0, 7.0])
    npt.assert_allclose(sens.sens_rsd.reshape(-1), expected_rsd)
    assert sens.egridname == "2G"


def test_collapse_rejects_inconsistent_weight_length():
    """Reject weights whose length does not match the fine-group count."""
    sens = _make_sensitivity([1.0, 2.0, 3.0, 4.0])

    with pytest.raises(SensitivityError, match="weighting function"):
        sens.collapse(fewgrp=[1.0, 3.0, 5.0], weight=np.array([1.0, 2.0]), )


def test_group_structure_converters_return_requested_unit():
    """Expose sensitivity energy boundaries in eV and MeV."""
    sens = _make_sensitivity([1.0, 2.0, 3.0, 4.0])

    npt.assert_allclose(sens.group_structure_mev, [1.0, 2.0, 3.0, 4.0, 5.0])
    npt.assert_allclose(sens.group_structure_ev, [1.0e6, 2.0e6, 3.0e6, 4.0e6, 5.0e6])
    assert sens.energy_grid.unit == "MeV"


def test_collapse_accepts_energy_grid_with_different_unit():
    """Convert the target collapse grid to the stored sensitivity unit."""
    sens = _make_sensitivity([1.0, 2.0, 3.0, 4.0])
    fewgrp = utils.EnergyGrid([1.0e6, 3.0e6, 5.0e6], unit="eV")

    sens.collapse(fewgrp=fewgrp)

    npt.assert_allclose(sens.group_structure, [1.0, 3.0, 5.0])
    assert sens.energy_unit == "MeV"
