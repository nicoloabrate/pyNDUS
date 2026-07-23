"""Tests for Covariance filesystem behavior."""

import importlib
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


class DummyErrorr:
    """Minimal ERRORR object exposing covariance metadata."""

    def __init__(self):
        """Populate MT and MAT metadata for one covariance section."""
        self.mt = [451, 18]
        self.mat = [1234]

    def get_cov(self):
        """Return an object with covariance data."""

        class Cov:
            """Container matching sandy Errorr.get_cov().data."""

            data = np.eye(1)

        return Cov()


class DummyEndf6:
    """Minimal ENDF-6 tape object used by Covariance tests."""

    @classmethod
    def from_file(cls, path):
        """Read an existing ENDF-6 file."""
        tape = cls()
        tape.read_path = path
        return tape

    def to_file(self, path):
        """Write a placeholder ENDF-6 file."""
        path.write_text("endf")


class DummyGeneratedErrorr:
    """Minimal generated ERRORR tape exposing to_file."""

    def to_file(self, path):
        """Write a placeholder generated ERRORR file."""
        path.write_text("generated errorr")


class DummyEndf6Processor:
    """Minimal ENDF-6 tape that emulates SANDY get_errorr calls."""

    def get_errorr(self,
                   *,
                   groupr_kws,
                   errorr_kws,
                   dryrun=False,
                   temperature=None,
                   verbose=True):
        """Return NJOY input text in dry-run mode and generated ERRORR tapes otherwise."""
        if dryrun:
            return "njoy input"
        return {"errorr33": DummyGeneratedErrorr()}


def _load_covariance_module(monkeypatch):
    """Import pyNDUS.covariance with a fake sandy module."""
    fake_sandy = SimpleNamespace(
        __version__="test",
        Endf6=DummyEndf6,
        Errorr=SimpleNamespace(from_file=lambda path: DummyErrorr()),
        get_endf6_file=lambda lib, kind, zaid: DummyEndf6(),
    )
    monkeypatch.setitem(sys.modules, "sandy", fake_sandy)
    sys.modules.pop("pyNDUS.covariance", None)
    return importlib.import_module("pyNDUS.covariance")


def test_non_database_mode_reads_errorr_files_from_cwd(monkeypatch, tmp_path):
    """Read existing ERRORR files directly from cwd when database is False."""
    module = _load_covariance_module(monkeypatch)
    (tmp_path / "U-235_300K.errorr33").write_text("errorr")

    cov = module.Covariance(
        922350,
        temperature=300,
        group_structure=[1.0, 2.0],
        egridname="custom",
        cwd=tmp_path,
        database=False,
    )

    assert cov.path == tmp_path
    assert set(cov.rcov) == {"errorr33"}


def test_non_database_mode_stores_mf35_covariance(monkeypatch, tmp_path):
    """Store MF35 covariance data when an ERRORR35 file is available."""
    module = _load_covariance_module(monkeypatch)
    (tmp_path / "U-235_300K.errorr35").write_text("errorr")

    cov = module.Covariance(
        922350,
        temperature=300,
        group_structure=[1.0, 2.0],
        egridname="custom",
        cwd=tmp_path,
        database=False,
    )

    assert set(cov.rcov) == {"errorr35"}


def test_get_returns_explicit_mf35_covariance(monkeypatch):
    """Return MF35 blocks when MF35 is explicitly requested."""
    module = _load_covariance_module(monkeypatch)
    cov = object.__new__(module.Covariance)
    cov._mat = 1234
    cov._MFs2MTs = {"errorr33": [18], "errorr35": [18]}

    energy = pd.IntervalIndex.from_breaks([1.0, 2.0, 3.0])
    index = pd.MultiIndex.from_tuples([(1234, 18, e) for e in energy],
                                      names=["MAT", "MT", "E"])
    mf35 = pd.DataFrame(np.diag([0.1, 0.2]), index=index, columns=index)

    cov._rcov = {
        "errorr33": pd.DataFrame(np.zeros((2, 2)), index=index, columns=index),
        "errorr35": mf35,
    }

    out = cov.get((18, 18), MF=35, to_numpy=True)

    np.testing.assert_allclose(out, np.diag([0.1, 0.2]))


def test_get_requires_explicit_mf(monkeypatch):
    """Reject covariance requests that do not identify an MF section."""
    module = _load_covariance_module(monkeypatch)
    cov = object.__new__(module.Covariance)

    with pytest.raises(TypeError, match="MF"):
        cov.get(18)

    with pytest.raises(TypeError):
        cov.get(18, 33)


def test_non_database_mode_generates_missing_errorr_files_in_cwd(
        monkeypatch, tmp_path):
    """Generate missing ERRORR files directly in cwd when database is False."""
    module = _load_covariance_module(monkeypatch)
    calls = {}

    def fake_sandy_calls_errorr(**kwargs):
        """Record generation arguments and return one ERRORR object."""
        calls.update(kwargs)
        (kwargs["errorr_dir"] /
         f"{kwargs['errorr_name']}33").write_text("errorr")
        return {"errorr33": DummyErrorr()}

    monkeypatch.setattr(
        module.Covariance,
        "sandy_calls_errorr",
        staticmethod(fake_sandy_calls_errorr),
    )

    cov = module.Covariance(
        922350,
        temperature=300,
        group_structure=[1.0e-6, 2.0e-6],
        egridname="custom",
        cwd=tmp_path,
        database=False,
        energy_unit="MeV",
    )

    assert cov.path == tmp_path
    assert (tmp_path / "U-235.endf").exists()
    assert (tmp_path / "U-235_300K.errorr33").exists()
    assert calls["errorr_dir"] == tmp_path
    assert np.allclose(calls["group_structure"], [1.0, 2.0])
    assert set(cov.rcov) == {"errorr33"}


def test_sandy_calls_errorr_uses_cwd_for_non_database_auxiliary_files(
        monkeypatch, tmp_path):
    """Write NJOY input/output folders beside flat ERRORR files."""
    module = _load_covariance_module(monkeypatch)
    monkeypatch.setenv("NJOY", "dummy-njoy")

    generated = module.Covariance.sandy_calls_errorr(
        endf6_tape=DummyEndf6Processor(),
        zaid=922350,
        temperature=300,
        group_structure=[1.0, 2.0],
        egridname="custom",
        errorr_dir=tmp_path,
        errorr_name="U-235_300K.errorr",
        process_resonances=True,
        lib="endfb_80",
        author="tester",
        njoy_ver="test-njoy",
    )

    assert set(generated) == {"errorr33"}
    assert (tmp_path / "U-235_300K.errorr33").exists()
    assert (tmp_path / "njoy_input" / "U-235_300K.input").exists()
    assert (tmp_path / "njoy_output" / "U-235_300K.log").exists()
