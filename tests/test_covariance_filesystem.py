"""Tests for Covariance filesystem behavior."""

import importlib
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _cleanup_fake_covariance_module():
    """Remove covariance modules imported with fake sandy after each test."""
    yield
    sys.modules.pop("pyNDUS.covariance", None)
    parent = sys.modules.get("pyNDUS")
    if parent is not None and hasattr(parent, "covariance"):
        delattr(parent, "covariance")


class DummyErrorr:
    """Minimal ERRORR object exposing covariance metadata."""

    def __init__(self, path=None):
        """Populate MT and MAT metadata for one covariance section."""
        self.read_path = path
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
    loaded_paths = []

    def errorr_from_file(path):
        loaded_paths.append(path)
        return DummyErrorr(path)

    fake_sandy = SimpleNamespace(
        __version__="test",
        Endf6=DummyEndf6,
        Errorr=SimpleNamespace(from_file=errorr_from_file),
        get_endf6_file=lambda lib, kind, zaid: DummyEndf6(),
        loaded_paths=loaded_paths,
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


@pytest.mark.parametrize(
    "filename",
    [
        "U-235_300K.errorr33",
        "U-235.errorr33",
        "9223500.errorr33",
        "922350.errorr33",
    ],
)
def test_database_mode_reads_supported_errorr_file_names(
        monkeypatch, tmp_path, filename):
    """Read legacy and temperature-free ERRORR names from database trees."""
    module = _load_covariance_module(monkeypatch)
    errorr_dir = tmp_path / "endfb_80" / "ECCO-33" / "errorr"
    errorr_dir.mkdir(parents=True)
    (errorr_dir / filename).write_text("errorr")

    def fail_sandy_calls_errorr(**kwargs):
        raise AssertionError("existing ERRORR files should not be regenerated")

    monkeypatch.setattr(
        module.Covariance,
        "sandy_calls_errorr",
        staticmethod(fail_sandy_calls_errorr),
    )

    cov = module.Covariance(
        922350,
        temperature=300,
        group_structure=[1.0, 2.0],
        egridname="ECCO-33",
        lib="endfb_80",
        cwd=tmp_path,
        database=True,
    )

    assert set(cov.rcov) == {"errorr33"}
    assert module.sandy.loaded_paths == [errorr_dir / filename]


def test_non_database_mode_reads_zaid0_errorr_name(monkeypatch, tmp_path):
    """Read Serpent-style ZAID0 ERRORR names directly from cwd."""
    module = _load_covariance_module(monkeypatch)
    path = tmp_path / "9223500.errorr35"
    path.write_text("errorr")

    cov = module.Covariance(
        922350,
        temperature=900,
        group_structure=[1.0, 2.0],
        egridname="custom",
        cwd=tmp_path,
        database=False,
    )

    assert set(cov.rcov) == {"errorr35"}
    assert module.sandy.loaded_paths == [path]


def test_legacy_temperature_errorr_name_has_lookup_priority(
        monkeypatch, tmp_path):
    """Prefer the historical pyNDUS filename when several names coexist."""
    module = _load_covariance_module(monkeypatch)
    legacy = tmp_path / "U-235_300K.errorr33"
    zaid = tmp_path / "922350.errorr33"
    legacy.write_text("legacy")
    zaid.write_text("zaid")

    cov = module.Covariance(
        922350,
        temperature=300,
        group_structure=[1.0, 2.0],
        egridname="custom",
        cwd=tmp_path,
        database=False,
    )

    assert module.sandy.loaded_paths == [legacy]


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


def test_mfs2mts_uses_covariance_dataframe_mts_before_metadata(monkeypatch):
    """Use stored matrix MTs instead of ERRORR metadata keys such as ZAIDs."""
    module = _load_covariance_module(monkeypatch)
    energy = pd.IntervalIndex.from_breaks([1.0, 2.0, 3.0])
    index = pd.MultiIndex.from_tuples([(2, 942390, e) for e in energy])

    class ErrorrWithFrame:
        """ERRORR object whose metadata would be misleading if iterated."""

        mt = {942390: [451, 251]}

        def get_cov(self):
            """Return covariance data stored as MF34/MT2."""
            return SimpleNamespace(data=pd.DataFrame(
                np.diag([0.1, 0.2]), index=index, columns=index))

    cov = object.__new__(module.Covariance)
    cov.MFs2MTs = {"errorr34": ErrorrWithFrame()}

    assert cov.MFs2MTs == {"errorr34": [2]}


def test_mfs2mts_drops_declared_mts_when_matrix_is_empty(monkeypatch):
    """Do not advertise declared MTs that have no covariance rows/columns."""
    module = _load_covariance_module(monkeypatch)
    empty_index = pd.MultiIndex.from_tuples([], names=["MAT", "MT", "E"])

    class EmptyErrorr:
        """ERRORR object with declared MTs but no extracted matrix data."""

        mt = [451, 251]

        def get_cov(self):
            """Return an empty covariance DataFrame."""
            return SimpleNamespace(
                data=pd.DataFrame(index=empty_index, columns=empty_index))

    cov = object.__new__(module.Covariance)
    cov.MFs2MTs = {"errorr34": EmptyErrorr()}

    assert cov.MFs2MTs == {"errorr34": []}


def test_get_accepts_mf34_mt251_alias_for_stored_mt2(monkeypatch):
    """Read MF34/MT2 covariance when the caller requests reduced MT251."""
    module = _load_covariance_module(monkeypatch)
    cov = object.__new__(module.Covariance)
    cov._mat = 1234
    cov._MFs2MTs = {"errorr34": [2]}

    energy = pd.IntervalIndex.from_breaks([1.0, 2.0, 3.0])
    index = pd.MultiIndex.from_tuples([(1234, 2, e) for e in energy],
                                      names=["MAT", "MT", "E"])
    cov._rcov = {
        "errorr34": pd.DataFrame(np.diag([0.3, 0.4]),
                                 index=index,
                                 columns=index),
    }

    out = cov.get((251, 251), MF=34, to_numpy=True)

    np.testing.assert_allclose(out, np.diag([0.3, 0.4]))


def test_get_requires_explicit_mf(monkeypatch):
    """Reject covariance requests that do not identify an MF section."""
    module = _load_covariance_module(monkeypatch)
    cov = object.__new__(module.Covariance)

    with pytest.raises(TypeError, match="MF"):
        cov.get(18)

    with pytest.raises(TypeError):
        cov.get(18, 33)


def _covariance_with_matrix(module, matrix, mf="errorr33"):
    """Build a Covariance instance around one labeled matrix."""
    cov = object.__new__(module.Covariance)
    index = pd.Index(range(len(matrix)), name="group")
    cov._rcov = {mf: pd.DataFrame(matrix, index=index, columns=index)}
    return cov


def test_check_psd_uses_extreme_eigenvalues_unless_full_spectrum_is_requested(
        monkeypatch):
    """Keep the ordinary check cheap and compute eigenpairs only on request."""
    module = _load_covariance_module(monkeypatch)
    cov = _covariance_with_matrix(module, [[1.0, 1.2], [1.2, 1.0]])

    diagnostics = cov.check_psd(dim_warning_threshold=None)
    full_diagnostics, eigenpairs = cov.check_psd(
        return_eigenpairs=True, dim_warning_threshold=None)

    assert not diagnostics.loc[33, "is_psd"]
    assert diagnostics.loc[33, "negative_eigenvalues"] is None
    assert full_diagnostics.loc[33, "negative_eigenvalues"] == 1
    assert eigenpairs["errorr33"][0].shape == (2, )
    assert eigenpairs["errorr33"][1].shape == (2, 2)


@pytest.mark.parametrize("method", ["clipped", "nearest"])
@pytest.mark.filterwarnings("ignore:statsmodels PSD repair")
def test_statsmodels_psd_repair_preserves_diagonal_and_original(monkeypatch, method):
    """Repair a complete MF without mutating the source covariance."""
    pytest.importorskip("statsmodels")
    module = _load_covariance_module(monkeypatch)
    matrix = np.array([[1.0, 1.2, 0.0],
                       [1.2, 1.0, 0.0],
                       [0.0, 0.0, 0.0]])
    cov = _covariance_with_matrix(module, matrix)

    repaired = cov.repair_psd(method=method, max_rel_correction=None,
                              dim_warning_threshold=None)

    np.testing.assert_allclose(cov.rcov["errorr33"], matrix)
    np.testing.assert_allclose(np.diag(repaired.rcov["errorr33"]),
                               np.diag(matrix))
    np.testing.assert_allclose(repaired.rcov["errorr33"].iloc[2], 0.0)
    assert not cov.was_repaired
    assert repaired.was_repaired
    assert repaired.is_psd(dim_warning_threshold=None)
    assert repaired.psd_history.loc[0, "scope"] == "complete_MF"
    assert repaired.psd_history.loc[0, "diagonal_preserved"]


def test_statsmodels_psd_repair_enforces_configurable_relative_limit(monkeypatch):
    """Reject a large repair unless the user explicitly disables the limit."""
    pytest.importorskip("statsmodels")
    module = _load_covariance_module(monkeypatch)
    cov = _covariance_with_matrix(module, [[1.0, 1.2], [1.2, 1.0]])

    with pytest.raises(module.CovarianceError, match="max_rel_correction"):
        cov.repair_psd(method="clipped", max_rel_correction=0.10,
                       dim_warning_threshold=None)

    repaired = cov.repair_psd(method="clipped", max_rel_correction=None,
                              dim_warning_threshold=None)
    assert repaired.is_psd(dim_warning_threshold=None)


def test_statsmodels_psd_repair_rejects_zero_variance_with_covariance(monkeypatch):
    """Reject a fixed diagonal for which no PSD covariance can exist."""
    pytest.importorskip("statsmodels")
    module = _load_covariance_module(monkeypatch)
    cov = _covariance_with_matrix(module, [[1.0, 0.1], [0.1, 0.0]])

    with pytest.raises(module.CovarianceError,
                       match="zero variance with non-zero covariance"):
        cov.repair_psd(method="clipped", max_rel_correction=None,
                       dim_warning_threshold=None)


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
