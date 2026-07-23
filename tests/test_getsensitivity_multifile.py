"""Tests for reading and merging multiple Serpent sensitivity files."""

from collections import OrderedDict
import importlib

import numpy as np
import numpy.testing as npt
import pytest

from pyNDUS.sensitivity import Sensitivity, SensitivityError

# Import del modulo, necessario per monkeypatchare st.read
getsensitivity_module = importlib.import_module("pyNDUS.sensitivity")


class FakeSerpentSensitivity:
    """
    Minimal mock of the object returned by serpentTools.read().

    The sensitivity arrays follow the shape expected by pyNDUS:

        (n_materials, n_za, n_mt, n_groups, 2)

    where the final dimension contains:
        [..., 0] = average sensitivity
        [..., 1] = relative standard deviation
    """

    def __init__(self, *, responses, materials, zaids, perturbations, energies, profiles, ):
        """Build a fake serpentTools sensitivity parser from sparse profiles."""
        self.materials = OrderedDict((material, i) for i, material in enumerate(materials))
        self.zais = OrderedDict((za, i) for i, za in enumerate(zaids))
        self.perts = OrderedDict((perturbation, i) for i, perturbation in enumerate(perturbations))
        self.energies = np.asarray(energies, dtype=float)

        n_mat = len(materials)
        n_za = len(zaids)
        n_mt = len(perturbations)
        n_groups = len(energies) - 1

        self.sensitivities = {}

        for response in responses:
            array = np.zeros((n_mat, n_za, n_mt, n_groups, 2), dtype=float, )

            for key, value in profiles.items():
                resp, material, za, perturbation = key

                if resp != response:
                    continue

                avg, rsd = value

                i_mat = self.materials[material]
                i_za = self.zais[za]
                i_mt = self.perts[perturbation]

                array[i_mat, i_za, i_mt, :, 0] = np.asarray(avg)
                array[i_mat, i_za, i_mt, :, 1] = np.asarray(rsd)

            self.sensitivities[response] = array


@pytest.fixture
def energy_grid():
    """Return a shared two-group energy grid for fake Serpent outputs."""
    return np.array([1.0e-5, 1.0e-3, 1.0])


def _make_fake_file(tmp_path, name):
    """Create an empty sensitivity file path accepted by the reader."""
    path = tmp_path / name
    path.touch()
    return path


def _extract_profile(sensitivity, response, material, za, mt):
    """Return flattened average and RSD profiles in input energy order."""
    avg, rsd = sensitivity.get(resp=[response], mat=[material], za=[za], MT=[mt],
                               group_order="ascending",
                               )

    return avg.reshape(-1), rsd.reshape(-1)


def test_multifile_merge_disjoint_profiles(tmp_path, monkeypatch, energy_grid, ):
    """Merge disjoint files and preserve each response, isotope, MT, and RSD."""
    file_1 = _make_fake_file(tmp_path, "case_1_sens0.m")
    file_2 = _make_fake_file(tmp_path, "case_2_sens0.m")

    parser_1 = FakeSerpentSensitivity(
        responses=["keff"], materials=["total"], zaids=[922350], perturbations=["xs 18"],
        energies=energy_grid, profiles={
            ("keff", "total", 922350, "xs 18"): ([1.0, 2.0], [0.01, 0.02],
                                                 ), },
    )

    parser_2 = FakeSerpentSensitivity(
        responses=["beff"], materials=["total"], zaids=[942390], perturbations=["xs 102"],
        energies=energy_grid, profiles={
            ("beff", "total", 942390, "xs 102"): ([3.0, 4.0], [0.03, 0.04],
                                                  ), },
    )

    parsers = {file_1: parser_1, file_2: parser_2, }

    monkeypatch.setattr(getsensitivity_module.st, "read", lambda path: parsers[path], )

    merged = Sensitivity([file_1, file_2], duplicate_policy="raise", )

    assert set(merged.responses) == {"keff", "beff"}
    assert set(merged.materials) == {"total"}
    assert set(merged.zaid) == {922350, 942390}
    assert set(merged.MTs) == {18, 102}

    avg_1, rsd_1 = _extract_profile(merged, "keff", "total", 922350, 18, )
    avg_2, rsd_2 = _extract_profile(merged, "beff", "total", 942390, 102, )

    npt.assert_allclose(avg_1, [1.0, 2.0])
    npt.assert_allclose(rsd_1, [0.01, 0.02])

    npt.assert_allclose(avg_2, [3.0, 4.0])
    npt.assert_allclose(rsd_2, [0.03, 0.04])


def test_multifile_inconsistent_energy_grid_raises(tmp_path, monkeypatch, ):
    """Reject a multifile merge when the Serpent energy grids differ."""
    file_1 = _make_fake_file(tmp_path, "case_1_sens0.m")
    file_2 = _make_fake_file(tmp_path, "case_2_sens0.m")

    parser_1 = FakeSerpentSensitivity(
        responses=["keff"], materials=["total"], zaids=[922350], perturbations=["xs 18"],
        energies=[1.0e-5, 1.0e-3, 1.0], profiles={
            ("keff", "total", 922350, "xs 18"): ([1.0, 2.0], [0.01, 0.02],
                                                 ), },
    )

    parser_2 = FakeSerpentSensitivity(
        responses=["beff"], materials=["total"], zaids=[942390], perturbations=["xs 102"],
        energies=[1.0e-5, 1.0e-4, 1.0], profiles={
            ("beff", "total", 942390, "xs 102"): ([3.0, 4.0], [0.03, 0.04],
                                                  ), },
    )

    parsers = {file_1: parser_1, file_2: parser_2, }

    monkeypatch.setattr(getsensitivity_module.st, "read", lambda path: parsers[path], )

    with pytest.raises(SensitivityError, match="Inconsistent energy grid", ):
        Sensitivity([file_1, file_2])


def test_multifile_duplicate_raise(tmp_path, monkeypatch, energy_grid, ):
    """Raise on duplicate response/material/ZA/MT profiles by default."""
    file_1 = _make_fake_file(tmp_path, "case_1_sens0.m")
    file_2 = _make_fake_file(tmp_path, "case_2_sens0.m")

    parser_1 = FakeSerpentSensitivity(
        responses=["keff"], materials=["total"], zaids=[922350], perturbations=["xs 18"],
        energies=energy_grid, profiles={
            ("keff", "total", 922350, "xs 18"): ([1.0, 2.0], [0.01, 0.02],
                                                 ), },
    )

    parser_2 = FakeSerpentSensitivity(
        responses=["keff"], materials=["total"], zaids=[922350], perturbations=["xs 18"],
        energies=energy_grid, profiles={
            ("keff", "total", 922350, "xs 18"): ([10.0, 20.0], [0.10, 0.20],
                                                 ), },
    )

    parsers = {file_1: parser_1, file_2: parser_2, }

    monkeypatch.setattr(getsensitivity_module.st, "read", lambda path: parsers[path], )

    with pytest.raises(SensitivityError, match="Duplicate sensitivity profile", ):
        Sensitivity([file_1, file_2], duplicate_policy="raise", )


def test_multifile_duplicate_keep_first(tmp_path, monkeypatch, energy_grid, ):
    """Keep the first duplicate profile when duplicate_policy is keep_first."""
    file_1 = _make_fake_file(tmp_path, "case_1_sens0.m")
    file_2 = _make_fake_file(tmp_path, "case_2_sens0.m")

    parser_1 = FakeSerpentSensitivity(
        responses=["keff"], materials=["total"], zaids=[922350], perturbations=["xs 18"],
        energies=energy_grid, profiles={
            ("keff", "total", 922350, "xs 18"): ([1.0, 2.0], [0.01, 0.02],
                                                 ), },
    )

    parser_2 = FakeSerpentSensitivity(
        responses=["keff"], materials=["total"], zaids=[922350], perturbations=["xs 18"],
        energies=energy_grid, profiles={
            ("keff", "total", 922350, "xs 18"): ([10.0, 20.0], [0.10, 0.20],
                                                 ), },
    )

    parsers = {file_1: parser_1, file_2: parser_2, }

    monkeypatch.setattr(getsensitivity_module.st, "read", lambda path: parsers[path], )

    merged = Sensitivity([file_1, file_2], duplicate_policy="keep_first", )

    avg, rsd = _extract_profile(merged, "keff", "total", 922350, 18, )

    npt.assert_allclose(avg, [1.0, 2.0])
    npt.assert_allclose(rsd, [0.01, 0.02])


def test_multifile_duplicate_keep_last(tmp_path, monkeypatch, energy_grid, ):
    """Keep the last duplicate profile when duplicate_policy is keep_last."""
    file_1 = _make_fake_file(tmp_path, "case_1_sens0.m")
    file_2 = _make_fake_file(tmp_path, "case_2_sens0.m")

    parser_1 = FakeSerpentSensitivity(
        responses=["keff"], materials=["total"], zaids=[922350], perturbations=["xs 18"],
        energies=energy_grid, profiles={
            ("keff", "total", 922350, "xs 18"): ([1.0, 2.0], [0.01, 0.02],
                                                 ), },
    )

    parser_2 = FakeSerpentSensitivity(
        responses=["keff"], materials=["total"], zaids=[922350], perturbations=["xs 18"],
        energies=energy_grid, profiles={
            ("keff", "total", 922350, "xs 18"): ([10.0, 20.0], [0.10, 0.20],
                                                 ), },
    )

    parsers = {file_1: parser_1, file_2: parser_2, }

    monkeypatch.setattr(getsensitivity_module.st, "read", lambda path: parsers[path], )

    merged = Sensitivity([file_1, file_2], duplicate_policy="keep_last", )

    avg, rsd = _extract_profile(merged, "keff", "total", 922350, 18, )

    npt.assert_allclose(avg, [10.0, 20.0])
    npt.assert_allclose(rsd, [0.10, 0.20])


@pytest.mark.parametrize("duplicate_policy", ["keep_first", "keep_last"], )
def test_same_file_twice_produces_object_identical_to_original(tmp_path, monkeypatch, energy_grid,
                                                               duplicate_policy,
                                                               ):
    """Merging the same file twice with a keep policy matches a single read."""
    file_path = _make_fake_file(tmp_path, "same_case_sens0.m", )

    parser = FakeSerpentSensitivity(
        responses=["keff", "beff"], materials=["total"], zaids=[922350, 942390],
        perturbations=["xs 18", "xs 102"], energies=energy_grid, profiles={
            ("keff", "total", 922350, "xs 18"): ([1.0, 2.0], [0.01, 0.02],
                                                 ),
            ("keff", "total", 942390, "xs 102"): ([3.0, 4.0], [0.03, 0.04],
                                                  ),
            ("beff", "total", 922350, "xs 18"): ([5.0, 6.0], [0.05, 0.06],
                                                 ),
            ("beff", "total", 942390, "xs 102"): ([7.0, 8.0], [0.07, 0.08],
                                                  ), },
    )

    monkeypatch.setattr(getsensitivity_module.st, "read", lambda path: parser, )

    original = Sensitivity(file_path)

    merged = Sensitivity([file_path, file_path], duplicate_policy=duplicate_policy, )

    assert merged.responses == original.responses
    assert merged.materials == original.materials
    assert merged.zaid == original.zaid
    assert merged.zais == original.zais
    assert merged.MTs == original.MTs

    npt.assert_array_equal(merged.group_structure, original.group_structure, )

    npt.assert_allclose(merged.sens, original.sens, rtol=0.0, atol=0.0, )

    npt.assert_allclose(merged.sens_rsd, original.sens_rsd, rtol=0.0, atol=0.0, )


def test_same_file_twice_raise_policy_detects_duplicates(tmp_path, monkeypatch, energy_grid, ):
    """Merging the same file twice with raise policy reports duplicates."""
    file_path = _make_fake_file(tmp_path, "same_case_sens0.m", )

    parser = FakeSerpentSensitivity(
        responses=["keff"], materials=["total"], zaids=[922350], perturbations=["xs 18"],
        energies=energy_grid, profiles={
            ("keff", "total", 922350, "xs 18"): ([1.0, 2.0], [0.01, 0.02],
                                                 ), },
    )

    monkeypatch.setattr(getsensitivity_module.st, "read", lambda path: parser, )

    with pytest.raises(SensitivityError, match="Duplicate sensitivity profile", ):
        Sensitivity([file_path, file_path], duplicate_policy="raise", )


def test_multifile_invalid_duplicate_policy_raises(tmp_path, monkeypatch, energy_grid, ):
    """Reject unknown duplicate-policy values before completing the merge."""
    file_path = _make_fake_file(tmp_path, "case_sens0.m", )

    parser = FakeSerpentSensitivity(
        responses=["keff"], materials=["total"], zaids=[922350], perturbations=["xs 18"],
        energies=energy_grid, profiles={
            ("keff", "total", 922350, "xs 18"): ([1.0, 2.0], [0.01, 0.02],
                                                 ), },
    )

    monkeypatch.setattr(getsensitivity_module.st, "read", lambda path: parser, )

    with pytest.raises(ValueError, match="Invalid duplicate_policy", ):
        Sensitivity([file_path, file_path], duplicate_policy="invalid", )
