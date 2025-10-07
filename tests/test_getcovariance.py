import pytest
from pathlib import Path
import numpy as np

from pyNDUS.GetCovariance import GetCovariance, GetCovarianceError

# Dummy utils for testing
class DummyUtils:
    @staticmethod
    def zaid2zais(zaid):
        return f"Dummy-{zaid}"

# Patch utils in GetCovariance for testing
import pyNDUS.GetCovariance as gc
gc.utils = DummyUtils

@pytest.fixture
def dummy_errorr_out():
    class DummyErrorr:
        def __init__(self):
            self.mt = [451, 452, 102]
            self.mat = [1234]
        def get_cov(self):
            class Cov:
                data = np.eye(2)
            return Cov()
    return {"errorr31": DummyErrorr(), "errorr33": DummyErrorr()}

def test_input_validation():
    # Invalid zaid
    with pytest.raises(ValueError):
        GetCovariance("not_an_int")
    # Invalid temperature
    with pytest.raises(ValueError):
        GetCovariance(922350, temperature="hot")
    # Invalid group_structure
    with pytest.raises(ValueError):
        GetCovariance(922350, group_structure=42)
    # Invalid egridname
    with pytest.raises(ValueError):
        GetCovariance(922350, egridname=123)
    # Invalid lib
    with pytest.raises(ValueError):
        GetCovariance(922350, lib=123)
    # Invalid cwd
    with pytest.raises(ValueError):
        GetCovariance(922350, cwd=123.45)

def test_setters_and_properties(dummy_errorr_out, monkeypatch):
    # Patch methods that require file system or external codes
    monkeypatch.setattr(GetCovariance, "_validate_input_args", staticmethod(lambda *a, **k: Path(".")))
    monkeypatch.setattr(GetCovariance, "sandy_calls_errorr", staticmethod(lambda *a, **k: dummy_errorr_out))
    monkeypatch.setattr("pyNDUS.GetCovariance.mynuclides", {922350: "U-235"})

    cov = GetCovariance(922350, temperature=300, group_structure=[1,2,3], egridname="test", lib="endfb_80", cwd=".")
    # Test zaid property
    assert cov.zaid == 922350
    # Test zais property
    assert cov.zais == "Dummy-922350"
    # Test temperature property
    assert cov.temperature == 300
    # Test group_structure property
    assert cov.group_structure == [1,2,3]
    # Test egridname property
    assert cov.egridname == "test"
    # Test MFs2MTs property
    assert isinstance(cov.MFs2MTs, dict)
    # Test mat property
    assert isinstance(cov.mat, int)
    # Test rcov property
    assert isinstance(cov.rcov, dict)

def test_setter_validation(monkeypatch, dummy_errorr_out):
    monkeypatch.setattr(GetCovariance, "_validate_input_args", staticmethod(lambda *a, **k: Path(".")))
    monkeypatch.setattr(GetCovariance, "sandy_calls_errorr", staticmethod(lambda *a, **k: dummy_errorr_out))
    monkeypatch.setattr("pyNDUS.GetCovariance.mynuclides", {922350: "U-235"})
    cov = GetCovariance(922350, temperature=300, group_structure=[1,2,3], egridname="test", lib="endfb_80", cwd=".")
    # Test temperature setter validation
    with pytest.raises(ValueError):
        cov.temperature = -10
    # Test zaid setter validation
    with pytest.raises(ValueError):
        cov.zaid = -1
    with pytest.raises(ValueError):
        cov.zaid = "not_an_int"
    # Test zais setter validation
    with pytest.raises(ValueError):
        cov.zais = 123
