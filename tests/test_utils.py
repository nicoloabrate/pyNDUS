"""Tests for isotope conversion and uncertainty helper utilities."""

import numpy as np
import pytest

from pyNDUS import utils


def test_zais2zaid_basic():
    """Convert a basic isotope label to pyNDUS ZAID form."""
    assert utils.zais2zaid("U-235") == 922350


def test_zaid2zais_basic():
    """Convert a basic pyNDUS ZAID value to isotope-label form."""
    assert utils.zaid2zais(922350) == "U-235"


def test_np2unp_preserves_nominal_values():
    """Preserve nominal values when building an uncertain array."""
    arr = np.array([1.0, 2.0])
    rsd = np.array([0.1, 0.2])

    out = utils.np2unp(arr, rsd)

    assert np.allclose([x.n for x in out], arr)


def test_np2unp_correct_std_scaling():
    """Scale absolute standard deviations from relative standard deviations."""
    arr = np.array([1.0])
    rsd = np.array([0.1])

    out = utils.np2unp(arr, rsd)

    assert np.isclose(out[0].s, 0.1)


def test_energy_grid_converts_mev_to_ev():
    """Convert energy group boundaries from MeV to eV."""
    grid = utils.EnergyGrid([1.0e-6, 1.0, 2.0], unit="MeV")

    assert grid.unit == "MeV"
    assert np.allclose(grid.ev, [1.0, 1.0e6, 2.0e6])


def test_energy_grid_converts_ev_to_mev():
    """Convert energy group boundaries from eV to MeV."""
    grid = utils.EnergyGrid([1.0, 1.0e6, 2.0e6], unit="eV")

    assert grid.unit == "eV"
    assert np.allclose(grid.mev, [1.0e-6, 1.0, 2.0])


def test_energy_grid_rejects_unknown_unit():
    """Reject unsupported energy-unit labels."""
    with pytest.raises(ValueError, match="Energy unit"):
        utils.EnergyGrid([1.0, 2.0], unit="keV")
