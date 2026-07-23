"""Tests for the public pyNDUS package interface."""

import inspect
import sys
from types import SimpleNamespace

import pytest


def test_public_sensitivity_import_returns_class():
    """Expose Sensitivity as a class from the top-level package."""
    from pyNDUS import Sensitivity
    from pyNDUS.sensitivity import Sensitivity as SensitivityClass

    assert Sensitivity is SensitivityClass
    assert inspect.isclass(Sensitivity)


def test_public_sandwich_import_returns_class():
    """Expose Sandwich as a class from the top-level package."""
    from pyNDUS import Sandwich
    from pyNDUS.sandwich import Sandwich as SandwichClass

    assert Sandwich is SandwichClass
    assert inspect.isclass(Sandwich)


def test_public_covariance_import_returns_class(monkeypatch):
    """Expose Covariance as a class from the top-level package."""
    monkeypatch.setitem(sys.modules, "sandy", SimpleNamespace(__version__="test"))
    sys.modules.pop("pyNDUS.covariance", None)

    import pyNDUS

    for name in ["Covariance", "CovarianceError"]:
        pyNDUS.__dict__.pop(name, None)

    from pyNDUS import Covariance, CovarianceError

    assert inspect.isclass(Covariance)
    assert inspect.isclass(CovarianceError)


def test_historical_covariance_name_is_not_public(monkeypatch):
    """Do not expose the old GetCovariance class alias."""
    monkeypatch.setitem(sys.modules, "sandy", SimpleNamespace(__version__="test"))

    import pyNDUS

    pyNDUS.__dict__.pop("GetCovariance", None)

    with pytest.raises(ImportError):
        from pyNDUS import GetCovariance
