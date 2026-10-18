"""Tests for Sandwich uncertainty propagation and missing covariance policies."""

import inspect

import pytest
import numpy as np
import numpy.testing as npt
from uncertainties import unumpy as unp

import pyNDUS.sandwich as sandwich_module
from pyNDUS.sandwich import (
    NonPositiveSemidefiniteError,
    NonPositiveSemidefiniteWarning,
    Sandwich,
)
from pyNDUS.sensitivity import Sensitivity
from pyNDUS.channels import SensitivityChannel


class FakeSensitivity:
    """
    Minimal object supporting what Sandwich.compute_uncertainty uses:
      - attributes: responses, materials (dict), MTs (dict), zaid (dict), n_groups, reader
      - method: get(...) supporting both keyword and positional calling styles
    """

    def __init__(self, data, n_groups=2, reader="serpent"):
        """Store sparse sensitivity vectors and expose pyNDUS-like metadata."""
        # data keyed by (resp, mat, za, mt) -> np.array shape (n_groups,)
        self._data = data
        self.n_groups = n_groups
        self.reader = reader

        self.responses = sorted({k[0] for k in data.keys()})
        self.materials = {k[1]: None for k in data.keys()}
        self.zaid = {k[2]: None for k in data.keys()}
        self.MTs = {k[3]: None for k in data.keys()}

        # match real Sensitivity API pattern used in Sandwich.__init__
        self.sens_rsd = None

    def get(self, *args, **kwargs):
        """Return the sensitivity vector addressed by pyNDUS-style arguments."""
        # Called in two styles in Sandwich.py:
        # 1) sens.get(resp=[resp], mat=[mat], MT=[mt], za=[za], group_order="ascending")
        # 2) sens.get([resp], [mat], [mt], [za], group_order="ascending")
        if args and len(args) >= 4:
            resp = args[0][0]
            mat = args[1][0]
            mt = args[2][0]
            za = args[3][0]
        else:
            resp = kwargs["resp"][0]
            mat = kwargs["mat"][0]
            mt = kwargs["MT"][0]
            za = kwargs["za"][0]

        return self._data[(resp, mat, za, mt)]


class FakeMCSensitivity(FakeSensitivity):
    """Fake sensitivity object returning average coefficients and RSDs."""

    def __init__(self, data, rsd, n_groups=2, reader="serpent"):
        """Store sparse sensitivity vectors and matching relative deviations."""
        super().__init__(data, n_groups=n_groups, reader=reader)
        self._rsd = rsd
        self.sens_rsd = True

    def get(self, *args, **kwargs):
        """Return the sensitivity vector and its relative standard deviation."""
        if args and len(args) >= 4:
            resp = args[0][0]
            mat = args[1][0]
            mt = args[2][0]
            za = args[3][0]
        else:
            resp = kwargs["resp"][0]
            mat = kwargs["mat"][0]
            mt = kwargs["MT"][0]
            za = kwargs["za"][0]
        key = (resp, mat, za, mt)
        return self._data[key], self._rsd[key]


class FakeCovZA:
    """
    Minimal covariance object:
      - attributes: rcov (dict with MF keys), mat (placeholder)
      - method: get((mt1,mt2), MF=..., to_numpy=True) -> np.ndarray
    """

    def __init__(self, mats_by_mf):
        """Store covariance blocks grouped by MF and MT pair."""
        # mats_by_mf: mf -> dict[(mt1,mt2)] = matrix
        self._mats_by_mf = mats_by_mf
        self.rcov = {mf: True for mf in mats_by_mf.keys()}
        self.mat = None  # accessed but not used downstream in compute_uncertainty

    def get(self, mt_pair, *, MF, to_numpy=False):
        """Return the covariance block for one MF and MT pair."""
        C = self._mats_by_mf[MF][mt_pair]
        return C


class FakeSensitivityForInit(FakeSensitivity):
    """Fake sensitivity object matching the mapping style used by Sandwich.__init__."""

    def __init__(self, data, n_groups=2, reader="serpent"):
        """Store fake sensitivity data with ZA/ZAIS integer indices."""
        super().__init__(data, n_groups=n_groups, reader=reader)
        zaids = sorted({k[2] for k in data.keys()})
        self.zaid = {za: iza for iza, za in enumerate(zaids)}
        self.zais = {f"ZA{za}": iza for iza, za in enumerate(zaids)}


class FakeCovZAForInit(FakeCovZA):
    """Fake covariance object exposing the MF-to-MT mapping used by Sandwich.__init__."""

    def __init__(self, mats_by_mf):
        """Store covariance blocks and expose available MTs by MF."""
        super().__init__(mats_by_mf)
        self.MFs2MTs = {
            mf: sorted({mt
                        for pair in mats
                        for mt in pair})
            for mf, mats in mats_by_mf.items()
        }


def _real_serpent_sensitivity(perturbations,
                              profiles,
                              za=922350,
                              material="fuel"):
    """Build a minimal real Sensitivity object with Serpent channel metadata."""
    sens = object.__new__(Sensitivity)
    sens.reader = "serpent"
    sens.responses = ["keff"]
    sens.materials = [material]
    sens.zaid = [za]
    sens.zais = sens.zaid.keys()
    sens.MTs = perturbations
    sens.energy_unit = "MeV"
    sens.group_structure = [1.0e-11, 1.0e-7, 1.0]

    data = np.zeros((1, 1, 1, len(sens.MTs), sens.n_groups))
    for key, profile in profiles.items():
        if isinstance(key, int):
            channel = SensitivityChannel.from_endf(
                average_MF=3,
                average_MT=key,
                name=f"xs {key}",
            )
        else:
            channel = SensitivityChannel.from_alias(key)
        data[0, 0, 0, sens.channels[channel], :] = profile
    sens._sens = data
    sens._sens_rsd = None
    return sens


def _get_df_value(df_matrix, resp, mat, za_label, mt_row, mt_col):
    """Read one uncertainty contribution from the wide result matrix."""
    # df_matrix index: (RESPONSE, MATERIAL, ZA, MT_row)
    return df_matrix.loc[(resp, mat, za_label, mt_row), mt_col]


def test_uncertainty_single_mt_diagonal():
    """Compute the diagonal sandwich product for one MT."""
    za = 922350
    za_label = "U235"
    za_dict = {za: za_label}

    mf = "errorr31"
    mt = 18

    S = np.array([0.5, -0.2])
    C = np.diag([0.1, 0.2])

    sens = FakeSensitivity({("keff", "fuel", za, mt): S}, n_groups=2)

    covmat = {za: FakeCovZA({mf: {(mt, mt): C}})}

    map_MF2MT = {za: {mf: [mt]}}

    df_matrix, _ = Sandwich.compute_uncertainty(
        sens=sens,
        covmat=covmat,
        list_resp=["keff"],
        list_mat=["fuel"],
        map_MF2MT=map_MF2MT,
        za_dict=za_dict,
        sens_MC=False,
        sigma=1,
    )

    expected = float(S.T @ C @ S)
    got = _get_df_value(df_matrix, "keff", "fuel", za_label, mt, mt)

    npt.assert_allclose(got, expected, rtol=1e-14, atol=0.0)


def test_uncertainty_treats_sensitivity_rsd_as_relative():
    """Convert sensitivity RSDs to absolute uncertainties before sandwiching."""
    za = 922350
    za_label = "U235"
    mt = 18
    mf = "errorr33"
    values = np.array([0.1, 0.2])
    rsd = np.array([0.5, 0.25])
    C = np.eye(2)

    sens = FakeMCSensitivity({("keff", "fuel", za, mt): values},
                             {("keff", "fuel", za, mt): rsd},
                             n_groups=2)
    covmat = {za: FakeCovZA({mf: {(mt, mt): C}})}

    df_matrix, _ = Sandwich.compute_uncertainty(
        sens=sens,
        covmat=covmat,
        list_resp=["keff"],
        list_mat=["fuel"],
        map_MF2MT={za: {
            mf: [mt]
        }},
        za_dict={za: za_label},
        sens_MC=True,
        sigma=1,
    )

    got = _get_df_value(df_matrix, "keff", "fuel", za_label, mt, mt)
    expected_nominal = float(values.T @ C @ values)
    expected_std = np.sqrt(np.sum((2 * values * np.abs(values) * rsd)**2))
    npt.assert_allclose(float(unp.nominal_values(got)), expected_nominal)
    npt.assert_allclose(float(unp.std_devs(got)), expected_std)


def test_uncertainty_uses_mf35_when_selected():
    """Propagate explicitly selected MF35 covariance data."""
    za = 922350
    za_label = "U235"
    za_dict = {za: za_label}

    mf = "errorr35"
    mt = 18

    S = np.array([0.5, -0.2])
    C = np.diag([0.1, 0.2])

    sens = FakeSensitivity({("keff", "fuel", za, mt): S}, n_groups=2)
    covmat = {za: FakeCovZA({mf: {(mt, mt): C}})}
    map_MF2MT = {za: {mf: [mt]}}

    df_matrix, _ = Sandwich.compute_uncertainty(
        sens=sens,
        covmat=covmat,
        list_resp=["keff"],
        list_mat=["fuel"],
        map_MF2MT=map_MF2MT,
        za_dict=za_dict,
        sens_MC=False,
        sigma=1,
    )

    expected = float(S.T @ C @ S)
    got = _get_df_value(df_matrix, "keff", "fuel", za_label, mt, mt)

    npt.assert_allclose(got, expected, rtol=1e-14, atol=0.0)


def test_sandwich_constructor_keeps_explicit_mf35(monkeypatch):
    """Keep MF35 in the Sandwich calculation when requested with list_MFs."""
    monkeypatch.setattr(sandwich_module, "Sensitivity", FakeSensitivityForInit)
    monkeypatch.setattr(sandwich_module, "Covariance", FakeCovZAForInit)

    za = 922350
    mt = 18
    mf = "errorr35"
    S = np.array([0.5, -0.2])
    C = np.diag([0.1, 0.2])

    sens = FakeSensitivityForInit({("keff", "fuel", za, mt): S}, n_groups=2)
    covmat = {za: FakeCovZAForInit({mf: {(mt, mt): C}})}

    sand = Sandwich(
        sens=sens,
        covmat=covmat,
        list_resp=["keff"],
        list_mat=["fuel"],
        list_MFs=[35],
        include_MF=True,
    )

    expected = float(S.T @ C @ S)
    got = sand.uncertainty.loc[("keff", "fuel", f"ZA{za}", 35, mt), mt]

    assert sand.MFs2MTs == {za: {mf: [mt]}}
    npt.assert_allclose(got, expected, rtol=1e-14, atol=0.0)


def test_mf35_mt18_uses_chi_prompt_not_fission_xs(monkeypatch):
    """Resolve MF35/MT18 to Serpent chi prompt, not the MF33 fission XS."""
    monkeypatch.setattr(sandwich_module, "Covariance", FakeCovZAForInit)

    za = 922350
    xs = np.array([100.0, 100.0])
    chi_prompt = np.array([0.5, -0.2])
    C = np.diag([0.1, 0.2])

    sens = _real_serpent_sensitivity(
        ["xs 18", "chi prompt"],
        {
            18: xs,
            "chi prompt": chi_prompt
        },
        za=za,
    )
    covmat = {za: FakeCovZAForInit({"errorr35": {(18, 18): C}})}

    sand = Sandwich(
        sens=sens,
        covmat=covmat,
        list_resp=["keff"],
        list_mat=["fuel"],
        list_za=[za],
        list_MTs=[18],
        list_MFs=[35],
        include_MF=True,
    )

    expected = float(chi_prompt.T @ C @ chi_prompt)
    got = sand.uncertainty.loc[("keff", "fuel", "U-235", 35, 18), 18]

    npt.assert_allclose(got, expected, rtol=1e-14, atol=0.0)


def test_mf34_mt251_uses_elastic_legendre_moment_1(monkeypatch):
    """Resolve reduced MF34/MT251 covariance to Serpent ela leg mom 1."""
    monkeypatch.setattr(sandwich_module, "Covariance", FakeCovZAForInit)

    za = 922350
    xs = np.array([100.0, 100.0])
    leg1 = np.array([0.3, 0.4])
    C = np.diag([0.5, 0.25])

    sens = _real_serpent_sensitivity(
        ["xs 2", "ela leg mom 1"],
        {
            2: xs,
            "ela leg mom 1": leg1
        },
        za=za,
    )
    covmat = {za: FakeCovZAForInit({"errorr34": {(251, 251): C}})}

    sand = Sandwich(
        sens=sens,
        covmat=covmat,
        list_resp=["keff"],
        list_mat=["fuel"],
        list_za=[za],
        list_MTs=[2],
        list_MFs=[34],
        include_MF=True,
    )

    expected = float(leg1.T @ C @ leg1)
    got = sand.uncertainty.loc[("keff", "fuel", "U-235", 34, 251), 251]

    assert sand.MFs2MTs == {za: {"errorr34": [251]}}
    npt.assert_allclose(got, expected, rtol=1e-14, atol=0.0)


def test_mf34_mt2_covariance_uses_legendre_p1_when_requested_as_mt251(
        monkeypatch):
    """Resolve stored MF34/MT2 covariance to the supported L=1 channel."""
    monkeypatch.setattr(sandwich_module, "Covariance", FakeCovZAForInit)

    za = 922350
    xs = np.array([100.0, 100.0])
    leg1 = np.array([0.3, 0.4])
    leg2 = np.array([10.0, 20.0])
    C = np.diag([0.5, 0.25])

    sens = _real_serpent_sensitivity(
        ["xs 2", "ela leg mom 1", "ela leg mom 2"],
        {
            2: xs,
            "ela leg mom 1": leg1,
            "ela leg mom 2": leg2,
        },
        za=za,
    )
    covmat = {za: FakeCovZAForInit({"errorr34": {(2, 2): C}})}

    sand = Sandwich(
        sens=sens,
        covmat=covmat,
        list_resp=["keff"],
        list_mat=["fuel"],
        list_za=[za],
        list_MTs=[251],
        list_MFs=[34],
        include_MF=True,
    )

    expected = float(leg1.T @ C @ leg1)
    got = sand.uncertainty.loc[("keff", "fuel", "U-235", 34, 2), 2]

    assert sand.MFs2MTs == {za: {"errorr34": [2]}}
    npt.assert_allclose(got, expected, rtol=1e-14, atol=0.0)


def test_invalid_covariance_mt_reports_parsing_problem(monkeypatch):
    """Reject ZAID-like values before they become fake sensitivity channels."""
    monkeypatch.setattr(sandwich_module, "Covariance", FakeCovZAForInit)

    za = 922350
    sens = _real_serpent_sensitivity(
        ["xs 18"],
        {18: np.array([1.0, 2.0])},
        za=za,
    )
    covmat = {
        za: FakeCovZAForInit({"errorr33": {
            (942390, 942390): np.eye(2)
        }})
    }

    with pytest.raises(sandwich_module.SandwichError, match="MAT/ZAID"):
        Sandwich(
            sens=sens,
            covmat=covmat,
            list_resp=["keff"],
            list_mat=["fuel"],
            list_za=[za],
            list_MTs=None,
            list_MFs=[33],
        )


def test_default_mt_selection_excludes_total_cross_section(monkeypatch):
    """Exclude MT=1 by default but retain it when explicitly selected."""
    monkeypatch.setattr(sandwich_module, "Sensitivity", FakeSensitivityForInit)
    monkeypatch.setattr(sandwich_module, "Covariance", FakeCovZAForInit)

    za = 922350
    sens = FakeSensitivityForInit(
        {
            ("keff", "fuel", za, 1): np.array([2.0, 0.0]),
            ("keff", "fuel", za, 18): np.array([1.0, 0.0]),
        }, )
    covmat = {
        za:
        FakeCovZAForInit({
            "errorr33": {
                (1, 1): np.diag([100.0, 0.0]),
                (18, 18): np.diag([4.0, 0.0]),
            }
        })
    }

    default = Sandwich(
        sens=sens,
        covmat=covmat,
        list_resp=["keff"],
        list_mat=["fuel"],
        list_za=[za],
        list_MFs=[33],
    )
    assert default.MFs2MTs == {za: {"errorr33": [18]}}
    npt.assert_allclose(
        default.uncertainty_standard_deviation.loc[("keff", "fuel"),
                                                   "variance"],
        4.0,
    )

    explicit = Sandwich(
        sens=sens,
        covmat=covmat,
        list_resp=["keff"],
        list_mat=["fuel"],
        list_za=[za],
        list_MTs=[1],
        list_MFs=[33],
    )
    assert explicit.MFs2MTs == {za: {"errorr33": [1]}}
    npt.assert_allclose(
        explicit.uncertainty_standard_deviation.loc[("keff", "fuel"),
                                                    "variance"],
        400.0,
    )


def test_serpent_total_zaid_without_covariance_is_ignored_by_sandwich(
        monkeypatch):
    """Retain ZAID=0 in sensitivities without treating it as a covariance ZAID."""
    monkeypatch.setattr(sandwich_module, "Sensitivity", FakeSensitivityForInit)
    monkeypatch.setattr(sandwich_module, "Covariance", FakeCovZAForInit)

    total_za = 0
    physical_za = 922350
    sens = FakeSensitivityForInit(
        {
            ("keff", "fuel", total_za, 18): np.array([100.0, 0.0]),
            ("keff", "fuel", physical_za, 18): np.array([1.0, 0.0]),
        }, )
    covmat = {
        physical_za:
        FakeCovZAForInit({
            "errorr33": {
                (18, 18): np.diag([4.0, 0.0])
            },
        })
    }

    sand = Sandwich(
        sens=sens,
        covmat=covmat,
        list_resp=["keff"],
        list_mat=["fuel"],
        list_MFs=[33],
    )

    assert total_za not in sand.za
    assert list(sand.za) == [physical_za]
    npt.assert_allclose(
        sand.uncertainty_standard_deviation.loc[("keff", "fuel"), "variance"],
        4.0,
    )


def test_uncertainty_mc_offdiagonal_passes_za_as_keyword(monkeypatch):
    """Use keyword arguments when reading off-diagonal MC sensitivities."""
    monkeypatch.setattr(sandwich_module, "Covariance", FakeCovZAForInit)

    za = 922350
    s18 = np.array([1.0, 2.0])
    s102 = np.array([0.5, -0.25])
    cov18 = np.diag([0.1, 0.2])
    cov102 = np.diag([0.3, 0.4])
    cov_cross = np.diag([0.01, -0.02])

    sens = _real_serpent_sensitivity(
        ["xs 18", "xs 102"],
        {
            18: s18,
            102: s102,
        },
        za=za,
    )
    sens._sens_rsd = np.zeros_like(sens.sens)
    covmat = {
        za:
        FakeCovZAForInit({
            "errorr33": {
                (18, 18): cov18,
                (102, 102): cov102,
                (18, 102): cov_cross,
                (102, 18): cov_cross,
            }
        })
    }

    sand = Sandwich(
        sens=sens,
        covmat=covmat,
        list_resp=["keff"],
        list_mat=["fuel"],
        list_za=[za],
        list_MTs=[18, 102],
        list_MFs=[33],
        include_MF=True,
    )

    expected = float(s18.T @ cov_cross @ s102)
    got = sand.uncertainty.loc[("keff", "fuel", "U-235", 33, 18), 102]

    npt.assert_allclose(got.nominal_value, expected, rtol=1e-14, atol=0.0)


def test_numeric_serpent_nubar_channels_are_mf1_not_mf3():
    """Keep numeric Serpent nubar perturbations mapped to MF1/MF31."""
    sens = object.__new__(Sensitivity)
    sens.reader = "serpent"
    sens.MTs = [452, 455, 456]

    total = SensitivityChannel.from_alias("nubar total")
    delayed = SensitivityChannel.from_alias("nubar delayed")
    prompt = SensitivityChannel.from_alias("nubar prompt")

    assert sens.endf_channels[total]["MF"] == 1
    assert sens.endf_channels[delayed]["MF"] == 1
    assert sens.endf_channels[prompt]["MF"] == 1
    assert sens.get_covariance_sensitivity_keys(31, 452) == [total]
    assert sens.get_covariance_sensitivity_keys(31, 455) == [delayed]
    assert sens.get_covariance_sensitivity_keys(31, 456) == [prompt]
    assert sens.get_covariance_sensitivity_keys(33, 452) == []
    assert sens.get_covariance_sensitivity_keys(33, 455) == []
    assert sens.get_covariance_sensitivity_keys(33, 456) == []


def test_similarity_without_covmat_uses_channels_not_mf_mt_cartesian_product():
    """Build MF/MT pairs from sensitivity channels, not from flat MT lists."""
    za = 922350
    sens = _real_serpent_sensitivity(
        ["xs 18", "chi prompt"],
        {
            18: np.array([1.0, 2.0]),
            "chi prompt": np.array([0.5, -0.2]),
        },
        za=za,
    )

    sand = Sandwich(
        sens=sens,
        sens2=sens,
        similarity=True,
        list_resp=["keff"],
        list_za=[za],
        list_MTs=[18],
        list_MFs=[31, 33, 35],
        include_MF=True,
    )

    assert sand.MFs2MTs == {
        za: {
            "errorr33": [18],
            "errorr35": [18],
        }
    }


def test_sensitivity_get_accepts_channels_and_rejects_ambiguous_mt():
    """MT-only selection is rejected when multiple channels share that MT."""
    za = 922350
    xs = np.array([1.0, 2.0])
    chi_prompt = np.array([0.5, -0.2])
    sens = _real_serpent_sensitivity(
        ["xs 18", "chi prompt"],
        {
            18: xs,
            "chi prompt": chi_prompt,
        },
        za=za,
    )

    chi_channel = SensitivityChannel.from_alias("chi prompt")
    avg = sens.get(resp=["keff"], mat=["fuel"], za=[za], channel=[chi_channel])

    npt.assert_allclose(avg.reshape(-1), chi_prompt)
    with pytest.raises(ValueError, match="ambiguous"):
        sens.get(resp=["keff"], mat=["fuel"], za=[za], MT=18)


def test_elastic_legendre_p1_accepts_mf34_mt251_and_mf34_mt2_with_l():
    """P1 can be represented by reduced MT251 or by complete MF34/MT2/L=1."""
    p1 = SensitivityChannel.from_alias("ela leg mom 1")

    assert p1 == SensitivityChannel.from_endf(covariance_MF=34,
                                              covariance_MT=251)
    assert p1 == SensitivityChannel.from_endf(covariance_MF=34,
                                              covariance_MT=2,
                                              L=1)
    assert p1.matches_covariance(34, 251)
    assert p1.matches_covariance(34, 2)

    with pytest.raises(ValueError, match="multiple sensitivity channels"):
        SensitivityChannel.from_endf(covariance_MF=34, covariance_MT=2)


def test_cross_section_aliases_resolve_to_descriptive_channel_names():
    """Legacy xs aliases are accepted but not used as canonical names."""
    fission = SensitivityChannel.from_alias("fission")

    assert SensitivityChannel.from_alias("xs 18") == fission
    assert SensitivityChannel.from_endf(average_MF=3, average_MT=18) == fission
    assert fission.name == "fission"


def test_from_endf_requires_mf_mt_pair_to_resolve_registered_channel():
    """An MT number alone is not enough to infer a registered channel."""
    mt_only = SensitivityChannel.from_endf(average_MT=18)
    chi_prompt = SensitivityChannel.from_alias("chi prompt")

    assert mt_only != chi_prompt
    assert mt_only.average_MF is None
    assert mt_only.average_MT == 18
    assert SensitivityChannel.from_endf(covariance_MF=35,
                                        covariance_MT=18) == chi_prompt


def test_uncertainty_signed_sqrt_and_total_standard_deviation(monkeypatch):
    """Expose signed roots per term and root the sum for the total deviation."""
    monkeypatch.setattr(sandwich_module, "Sensitivity", FakeSensitivityForInit)
    monkeypatch.setattr(sandwich_module, "Covariance", FakeCovZAForInit)

    za = 922350
    mt1, mt2 = 18, 102
    mf = "errorr33"
    S1 = np.array([1.0, 0.0])
    S2 = np.array([1.0, 0.0])
    C11 = np.diag([4.0, 0.0])
    C22 = np.diag([9.0, 0.0])
    C12 = np.diag([-4.0, 0.0])
    C21 = np.diag([-4.0, 0.0])

    sens = FakeSensitivityForInit(
        {
            ("keff", "fuel", za, mt1): S1,
            ("keff", "fuel", za, mt2): S2,
        },
        n_groups=2,
    )
    covmat = {
        za:
        FakeCovZAForInit({
            mf: {
                (mt1, mt1): C11,
                (mt1, mt2): C12,
                (mt2, mt1): C21,
                (mt2, mt2): C22,
            }
        })
    }

    sand = Sandwich(
        sens=sens,
        covmat=covmat,
        list_resp=["keff"],
        list_mat=["fuel"],
        list_za=[za],
        list_MTs=[mt1, mt2],
        list_MFs=[33],
        include_MF=True,
        uncertainty_output="signed_sqrt",
    )

    label = f"ZA{za}"
    expected_signed = {
        (mt1, mt1): 2.0,
        (mt1, mt2): -2.0,
        (mt2, mt1): -2.0,
        (mt2, mt2): 3.0,
    }
    for (mt_row, mt_col), expected in expected_signed.items():
        got = sand.uncertainty.loc[("keff", "fuel", label, 33, mt_row), mt_col]
        npt.assert_allclose(got, expected, rtol=1e-14, atol=0.0)

    variance_12 = sand.uncertainty_variance.loc[("keff", "fuel", label, 33,
                                                 mt1), mt2]
    npt.assert_allclose(variance_12, -4.0, rtol=1e-14, atol=0.0)

    summary = sand.uncertainty_standard_deviation.loc[("keff", "fuel")]
    npt.assert_allclose(summary["variance"], 5.0, rtol=1e-14, atol=0.0)
    npt.assert_allclose(summary["standard_deviation"],
                        np.sqrt(5.0),
                        rtol=1e-14,
                        atol=0.0)


def test_uncertainty_output_rejects_unknown_representation(monkeypatch):
    """Reject uncertainty output representations that are not documented."""
    monkeypatch.setattr(sandwich_module, "Sensitivity", FakeSensitivityForInit)
    monkeypatch.setattr(sandwich_module, "Covariance", FakeCovZAForInit)

    za = 922350
    mt = 18
    mf = "errorr33"
    sens = FakeSensitivityForInit(
        {("keff", "fuel", za, mt): np.array([1.0, 0.0])},
        n_groups=2,
    )
    covmat = {za: FakeCovZAForInit({mf: {(mt, mt): np.eye(2)}})}

    with pytest.raises(ValueError, match="uncertainty_output"):
        Sandwich(
            sens=sens,
            covmat=covmat,
            list_resp=["keff"],
            list_mat=["fuel"],
            list_za=[za],
            list_MTs=[mt],
            list_MFs=[33],
            uncertainty_output="square_root",
        )


def test_uncertainty_two_mts_with_cross_terms():
    """Compute off-diagonal MT covariance contributions."""
    za = 922350
    za_label = "U235"
    za_dict = {za: za_label}

    mf = "errorr31"
    mt1, mt2 = 18, 102

    S1 = np.array([0.5, -0.2])
    S2 = np.array([0.4, -0.1])

    C11 = np.diag([0.1, 0.2])
    C22 = np.diag([0.3, 0.4])
    C12 = np.diag([0.01, 0.02])
    C21 = np.diag([0.01, 0.02])

    sens = FakeSensitivity(
        {
            ("keff", "fuel", za, mt1): S1,
            ("keff", "fuel", za, mt2): S2,
        },
        n_groups=2,
    )

    covmat = {
        za:
        FakeCovZA({
            mf: {
                (mt1, mt1): C11,
                (mt2, mt2): C22,
                (mt1, mt2): C12,
                (mt2, mt1): C21,
            }
        })
    }

    map_MF2MT = {za: {mf: [mt1, mt2]}}

    df_matrix, _ = Sandwich.compute_uncertainty(
        sens=sens,
        covmat=covmat,
        list_resp=["keff"],
        list_mat=["fuel"],
        map_MF2MT=map_MF2MT,
        za_dict=za_dict,
        sens_MC=False,
        sigma=1,
    )

    expected_12 = float(S1.T @ C12 @ S2)
    expected_21 = float(S2.T @ C21 @ S1)

    got_12 = _get_df_value(df_matrix, "keff", "fuel", za_label, mt1, mt2)
    got_21 = _get_df_value(df_matrix, "keff", "fuel", za_label, mt2, mt1)

    npt.assert_allclose(got_12, expected_12, rtol=1e-14, atol=0.0)
    npt.assert_allclose(got_21, expected_21, rtol=1e-14, atol=0.0)


def test_uncertainty_missing_covariance_zero_policy_returns_zero():
    """Return zero contribution when missing covariance uses zero policy."""
    za = 922350
    za_label = "U235"
    za_dict = {za: za_label}

    mf = "errorr31"
    mt = 18

    S = np.array([1.0, 2.0])

    sens = FakeSensitivity({("keff", "fuel", za, mt): S}, n_groups=2)

    covmat = {}  # ZA missing -> missing covariance path

    map_MF2MT = {za: {mf: [mt]}}

    df_matrix, _ = Sandwich.compute_uncertainty(
        sens=sens,
        covmat=covmat,
        list_resp=["keff"],
        list_mat=["fuel"],
        map_MF2MT=map_MF2MT,
        za_dict=za_dict,
        sens_MC=False,
        sigma=1,
        missing_cov="zero",
        missing_cov_rsd=0.20,
    )

    got = _get_df_value(df_matrix, "keff", "fuel", za_label, mt, mt)
    npt.assert_allclose(got, 0.0, rtol=0.0, atol=0.0)


def test_uncertainty_missing_covariance_raise_policy_raises():
    """Raise immediately when missing covariance uses raise policy."""
    za = 922350
    za_dict = {za: "U235"}

    mf = "errorr31"
    mt = 18

    S = np.array([1.0, 2.0])
    sens = FakeSensitivity({("keff", "fuel", za, mt): S}, n_groups=2)

    covmat = {}  # missing covariance
    map_MF2MT = {za: {mf: [mt]}}

    with pytest.raises(ValueError, match="Missing covariance"):
        Sandwich.compute_uncertainty(
            sens=sens,
            covmat=covmat,
            list_resp=["keff"],
            list_mat=["fuel"],
            map_MF2MT=map_MF2MT,
            za_dict=za_dict,
            sens_MC=False,
            sigma=1,
            missing_cov="raise",
            missing_cov_rsd=0.20,
        )


def test_uncertainty_missing_covariance_assume_policy_diagonal():
    """Use fallback diagonal covariance when missing covariance is assumed."""
    za = 922350
    za_label = "U235"
    za_dict = {za: za_label}

    mf = "errorr31"
    mt = 18

    S = np.array([1.0, 2.0])
    sens = FakeSensitivity({("keff", "fuel", za, mt): S}, n_groups=2)

    covmat = {}  # missing covariance
    map_MF2MT = {za: {mf: [mt]}}

    rsd = 0.20
    expected = float(S.T @ ((rsd**2) * np.eye(2)) @ S)

    df_matrix, _ = Sandwich.compute_uncertainty(
        sens=sens,
        covmat=covmat,
        list_resp=["keff"],
        list_mat=["fuel"],
        map_MF2MT=map_MF2MT,
        za_dict=za_dict,
        sens_MC=False,
        sigma=1,
        missing_cov="assume",
        missing_cov_rsd=rsd,
        missing_cov_corr=0.0,
    )

    got = _get_df_value(df_matrix, "keff", "fuel", za_label, mt, mt)
    npt.assert_allclose(got, expected, rtol=1e-14, atol=0.0)


def test_uncertainty_missing_covariance_assume_policy_offdiagonal_zero_corr():
    """Use zero off-diagonal fallback covariance when correlation is zero."""
    za = 922350
    za_label = "U235"
    za_dict = {za: za_label}

    mf = "errorr31"
    mt1, mt2 = 18, 102

    S1 = np.array([0.5, -0.2])
    S2 = np.array([0.4, -0.1])

    sens = FakeSensitivity(
        {
            ("keff", "fuel", za, mt1): S1,
            ("keff", "fuel", za, mt2): S2,
        },
        n_groups=2,
    )

    covmat = {}  # missing covariance
    map_MF2MT = {za: {mf: [mt1, mt2]}}

    df_matrix, _ = Sandwich.compute_uncertainty(
        sens=sens,
        covmat=covmat,
        list_resp=["keff"],
        list_mat=["fuel"],
        map_MF2MT=map_MF2MT,
        za_dict=za_dict,
        sens_MC=False,
        sigma=1,
        missing_cov="assume",
        missing_cov_rsd=0.20,
        missing_cov_corr=0.0,
    )

    got_12 = _get_df_value(df_matrix, "keff", "fuel", za_label, mt1, mt2)
    got_21 = _get_df_value(df_matrix, "keff", "fuel", za_label, mt2, mt1)

    npt.assert_allclose(got_12, 0.0, rtol=0.0, atol=0.0)
    npt.assert_allclose(got_21, 0.0, rtol=0.0, atol=0.0)


def test_uncertainty_missing_covariance_assume_policy_offdiagonal_nonzero_corr(
):
    """Use nonzero correlated fallback covariance for off-diagonal MT pairs."""
    za = 922350
    za_label = "U235"
    za_dict = {za: za_label}

    mf = "errorr31"
    mt1, mt2 = 18, 102

    S1 = np.array([0.5, -0.2])
    S2 = np.array([0.4, -0.1])

    sens = FakeSensitivity(
        {
            ("keff", "fuel", za, mt1): S1,
            ("keff", "fuel", za, mt2): S2,
        },
        n_groups=2,
    )

    covmat = {}
    map_MF2MT = {za: {mf: [mt1, mt2]}}

    rsd = 0.20
    corr = 0.5
    C_fb = corr * (rsd**2) * np.eye(2)

    expected_12 = float(S1.T @ C_fb @ S2)
    expected_21 = float(S2.T @ C_fb @ S1)

    df_matrix, _ = Sandwich.compute_uncertainty(
        sens=sens,
        covmat=covmat,
        list_resp=["keff"],
        list_mat=["fuel"],
        map_MF2MT=map_MF2MT,
        za_dict=za_dict,
        sens_MC=False,
        sigma=1,
        missing_cov="assume",
        missing_cov_rsd=rsd,
        missing_cov_corr=corr,
    )

    got_12 = _get_df_value(df_matrix, "keff", "fuel", za_label, mt1, mt2)
    got_21 = _get_df_value(df_matrix, "keff", "fuel", za_label, mt2, mt1)

    npt.assert_allclose(got_12, expected_12, rtol=1e-14, atol=0.0)
    npt.assert_allclose(got_21, expected_21, rtol=1e-14, atol=0.0)


def test_uncertainty_invalid_missing_cov_policy_raises():
    """Reject unsupported missing-covariance policies."""
    za = 922350
    za_dict = {za: "U235"}

    mf = "errorr31"
    mt = 18

    S = np.array([1.0, 2.0])
    sens = FakeSensitivity({("keff", "fuel", za, mt): S}, n_groups=2)

    covmat = {}
    map_MF2MT = {za: {mf: [mt]}}

    with pytest.raises(ValueError, match="Invalid missing_cov policy"):
        Sandwich.compute_uncertainty(
            sens=sens,
            covmat=covmat,
            list_resp=["keff"],
            list_mat=["fuel"],
            map_MF2MT=map_MF2MT,
            za_dict=za_dict,
            sens_MC=False,
            sigma=1,
            missing_cov="banana",
            missing_cov_rsd=0.20,
        )


def test_uncertainty_missing_sensitivity_returns_zero():
    """Return zero contribution when the requested sensitivity profile is absent."""
    za = 922350
    za_label = "U235"
    za_dict = {za: za_label}

    mf = "errorr31"
    mt = 18

    # Sensitivity object intentionally does NOT contain (keff, fuel, za, mt)
    sens = FakeSensitivity(
        {
            # Put a different MT so sens.MTs has at least one key and the object is non-empty
            ("keff", "fuel", za, 102):
            np.array([0.1, 0.2]),
        },
        n_groups=2,
    )

    # Covariance exists for the requested MT
    C = np.diag([0.1, 0.2])
    covmat = {za: FakeCovZA({mf: {(mt, mt): C}})}

    # Mapping requests MT=18, but sens does not provide it -> it should yield 0
    map_MF2MT = {za: {mf: [mt]}}

    df_matrix, _ = Sandwich.compute_uncertainty(
        sens=sens,
        covmat=covmat,
        list_resp=["keff"],
        list_mat=["fuel"],
        map_MF2MT=map_MF2MT,
        za_dict=za_dict,
        sens_MC=False,
        sigma=1,
    )

    got = _get_df_value(df_matrix, "keff", "fuel", za_label, mt, mt)
    npt.assert_allclose(got, 0.0, rtol=0.0, atol=0.0)


def _indefinite_two_mt_case(material="fuel"):
    """Return a two-group, two-MT covariance with negative eigenvalues."""
    za = 922380
    mf = "errorr33"
    mt1, mt2 = 2, 18
    sensitivities = FakeSensitivity(
        {
            ("keff", material, za, mt1): np.array([1.0, 0.0]),
            ("keff", material, za, mt2): np.array([0.0, 0.0]),
        },
        n_groups=2,
        reader="serpent",
    )
    blocks = {
        (mt1, mt1): np.eye(2),
        (mt1, mt2): 2.0 * np.eye(2),
        (mt2, mt1): 2.0 * np.eye(2),
        (mt2, mt2): np.eye(2),
    }
    covariance = {za: FakeCovZA({mf: blocks})}
    mapping = {za: {mf: [mt1, mt2]}}
    return sensitivities, covariance, mapping, {za: "U238"}


def test_non_psd_warn_preserves_covariance_and_exposes_modal_diagnostics():
    """Warn once, retain the raw result, and close every modal MT split."""
    sens, covmat, mapping, za_dict = _indefinite_two_mt_case()

    with pytest.warns(NonPositiveSemidefiniteWarning) as caught:
        result, _ = Sandwich.compute_uncertainty(sens,
                                                 covmat, ["keff"], ["fuel"],
                                                 mapping,
                                                 za_dict,
                                                 sens_MC=False,
                                                 non_psd_policy="warn")

    assert len(caught) == 1
    npt.assert_allclose(result.to_numpy(dtype=float).sum(), 1.0)
    diagnostics = result.attrs["covariance_diagnostics"]
    assert diagnostics.loc[("U238", 33), "negative_eigenvalues"] == 2
    assert not diagnostics.loc[("U238", 33), "repaired"]

    modes = result.attrs["eigenmodes"]
    mt_modes = result.attrs["eigenmode_mts"]
    npt.assert_allclose(modes["raw_variance"].sum(), 1.0)
    npt.assert_allclose(modes["used_variance"].sum(), 1.0)
    attributed = mt_modes["raw_attributed_variance"].groupby(
        level=["RESPONSE", "MATERIAL", "ZA", "MF", "EIGENMODE"]).sum()
    npt.assert_allclose(
        np.asarray(attributed, dtype=float),
        np.asarray(modes["raw_variance"], dtype=float),
    )


def test_non_psd_raise_identifies_the_selected_covariance():
    """Reject a non-PSD selected ZA/MF matrix before propagation."""
    sens, covmat, mapping, za_dict = _indefinite_two_mt_case()

    with pytest.raises(NonPositiveSemidefiniteError,
                       match=r"ZA=922380, MF=33, MTs=\(2, 18\)"):
        Sandwich.compute_uncertainty(sens,
                                     covmat, ["keff"], ["fuel"],
                                     mapping,
                                     za_dict,
                                     sens_MC=False,
                                     non_psd_policy="raise")


def test_non_psd_default_policy_is_raise():
    """Use the fail-safe raise policy by default in every public entry point."""
    assert inspect.signature(Sandwich).parameters[
        "non_psd_policy"].default == "raise"
    assert inspect.signature(Sandwich.compute_uncertainty).parameters[
        "non_psd_policy"].default == "raise"
    assert inspect.signature(Sandwich.compute_representativity).parameters[
        "non_psd_policy"].default == "raise"
    assert inspect.signature(Sandwich).parameters[
        "neg_eigval_policy"].default == "zero"
    assert inspect.signature(Sandwich.compute_uncertainty).parameters[
        "neg_eigval_policy"].default == "zero"
    assert inspect.signature(Sandwich.compute_representativity).parameters[
        "neg_eigval_policy"].default == "zero"

    sens, covmat, mapping, za_dict = _indefinite_two_mt_case()
    with pytest.raises(NonPositiveSemidefiniteError):
        Sandwich.compute_uncertainty(
            sens,
            covmat, ["keff"], ["fuel"],
            mapping,
            za_dict,
            sens_MC=False)


def test_non_psd_eigen_decomp_clips_modes_and_closes_to_uncertainty():
    """Propagate the reconstructed PSD matrix and retain raw diagnostics."""
    sens, covmat, mapping, za_dict = _indefinite_two_mt_case()

    with pytest.warns(NonPositiveSemidefiniteWarning) as caught:
        result, _ = Sandwich.compute_uncertainty(sens,
                                                 covmat, ["keff"], ["fuel"],
                                                 mapping,
                                                 za_dict,
                                                 sens_MC=False,
                                                 non_psd_policy="eigen_decomp")

    assert len(caught) == 1
    # C+ = 1.5 * [[1, 1], [1, 1]] and S = [1, 0].
    npt.assert_allclose(result.to_numpy(dtype=float).sum(), 1.5)
    modes = result.attrs["eigenmodes"]
    assert modes["negative"].sum() == 2
    assert modes["set_to_zero"].sum() == 2
    assert modes["absolute_value_applied"].sum() == 0
    npt.assert_allclose(modes["raw_variance"].sum(), 1.0)
    npt.assert_allclose(modes["used_variance"].sum(), 1.5)
    diagnostics = result.attrs["covariance_diagnostics"]
    assert diagnostics.loc[("U238", 33),
                           "negative_eigenvalues_set_to_zero"] == 2
    assert diagnostics.loc[("U238", 33), "absolute_eigenvalues"] == 0
    assert diagnostics.loc[("U238", 33),
                           "neg_eigval_policy"] == "zero"

    mt_modes = result.attrs["eigenmode_mts"]
    attributed = mt_modes["used_attributed_variance"].groupby(
        level=["RESPONSE", "MATERIAL", "ZA", "MF", "EIGENMODE"]).sum()
    npt.assert_allclose(
        np.asarray(attributed, dtype=float),
        np.asarray(modes["used_variance"], dtype=float),
    )


def test_non_psd_eigen_decomp_absolute_values_modes_and_closes_to_uncertainty():
    """Propagate absolute eigenvalues while retaining raw negative diagnostics."""
    sens, covmat, mapping, za_dict = _indefinite_two_mt_case()

    with pytest.warns(NonPositiveSemidefiniteWarning,
                      match="replaced by their absolute values") as caught:
        result, _ = Sandwich.compute_uncertainty(
            sens,
            covmat, ["keff"], ["fuel"],
            mapping,
            za_dict,
            sens_MC=False,
            non_psd_policy="eigen_decomp",
            neg_eigval_policy="absolute")

    assert len(caught) == 1
    # Q @ abs(Lambda) @ Q.T has diagonal 2 and S = [1, 0].
    npt.assert_allclose(result.to_numpy(dtype=float).sum(), 2.0)
    modes = result.attrs["eigenmodes"]
    negative_modes = modes.loc[modes["negative"]]
    assert modes["set_to_zero"].sum() == 0
    assert modes["absolute_value_applied"].sum() == 2
    npt.assert_allclose(negative_modes["used_eigenvalue"],
                        -negative_modes["eigenvalue"])
    npt.assert_allclose(negative_modes["used_variance"],
                        -negative_modes["raw_variance"])
    npt.assert_allclose(modes["raw_variance"].sum(), 1.0)
    npt.assert_allclose(modes["used_variance"].sum(), 2.0)
    diagnostics = result.attrs["covariance_diagnostics"]
    assert diagnostics.loc[("U238", 33),
                           "negative_eigenvalues_set_to_zero"] == 0
    assert diagnostics.loc[("U238", 33), "absolute_eigenvalues"] == 2
    assert diagnostics.loc[("U238", 33),
                           "neg_eigval_policy"] == "absolute"

    mt_modes = result.attrs["eigenmode_mts"]
    attributed = mt_modes["used_attributed_variance"].groupby(
        level=["RESPONSE", "MATERIAL", "ZA", "MF", "EIGENMODE"]).sum()
    npt.assert_allclose(
        np.asarray(attributed, dtype=float),
        np.asarray(modes["used_variance"], dtype=float),
    )


@pytest.mark.parametrize(("neg_eigval_policy", "expected"),
                         [("zero", 1.0), ("absolute", 0.5)])
def test_representativity_eigen_decomp_reuses_psd_matrix_in_all_terms(
        neg_eigval_policy, expected):
    """Use one repaired matrix for numerator and both denominator variances."""
    sens_a, covmat, mapping, za_dict = _indefinite_two_mt_case(
        material="total")
    za = next(iter(za_dict))
    sens_b = FakeSensitivity(
        {
            ("keff", "total", za, 2): np.array([0.0, 0.0]),
            ("keff", "total", za, 18): np.array([1.0, 0.0]),
        },
        n_groups=2,
        reader="serpent",
    )

    with pytest.warns(NonPositiveSemidefiniteWarning) as caught:
        result, _ = Sandwich.compute_representativity(
            sens_a,
            sens_b,
            covmat, ["keff"],
            mapping,
            za_dict,
            sens_MC=False,
            non_psd_policy="eigen_decomp",
            neg_eigval_policy=neg_eigval_policy)

    # Preparing once also ensures that one diagnostic warning is emitted.
    assert len(caught) == 1
    npt.assert_allclose(result.to_numpy(dtype=float).sum(), expected)
    diagnostics = result.attrs["covariance_diagnostics"]
    assert diagnostics.loc[("U238", 33), "repaired"]


def test_invalid_non_psd_policy_is_rejected():
    """Reject unknown non-PSD policies explicitly."""
    sens, covmat, mapping, za_dict = _indefinite_two_mt_case()

    with pytest.raises(ValueError, match="non_psd_policy"):
        Sandwich.compute_uncertainty(sens,
                                     covmat, ["keff"], ["fuel"],
                                     mapping,
                                     za_dict,
                                     sens_MC=False,
                                     non_psd_policy="clip")


def test_invalid_neg_eigval_policy_is_rejected():
    """Reject unknown negative-eigenvalue replacements explicitly."""
    sens, covmat, mapping, za_dict = _indefinite_two_mt_case()

    with pytest.raises(ValueError, match="neg_eigval_policy"):
        Sandwich.compute_uncertainty(
            sens,
            covmat, ["keff"], ["fuel"],
            mapping,
            za_dict,
            sens_MC=False,
            non_psd_policy="eigen_decomp",
            neg_eigval_policy="reflect")


def test_constructor_exposes_eigen_decomp_diagnostics(monkeypatch):
    """Expose PSD and modal diagnostics on a regular Sandwich result."""
    monkeypatch.setattr(sandwich_module, "Sensitivity",
                        FakeSensitivityForInit)
    monkeypatch.setattr(sandwich_module, "Covariance", FakeCovZAForInit)
    sens, covmat, mapping, _ = _indefinite_two_mt_case()
    za = next(iter(mapping))
    sens = FakeSensitivityForInit(sens._data, n_groups=2, reader="serpent")
    covmat = {
        za: FakeCovZAForInit(covmat[za]._mats_by_mf),
    }

    with pytest.warns(NonPositiveSemidefiniteWarning):
        result = Sandwich(
            sens,
            covmat=covmat,
            list_za=[za],
            list_MFs=[33],
            list_MTs=[2, 18],
            non_psd_policy="eigen_decomp",
            neg_eigval_policy="absolute",
        )

    npt.assert_allclose(
        result.uncertainty_standard_deviation.loc[("keff", "fuel"),
                                                  "variance"],
        2.0,
    )
    assert result.non_psd_policy == "eigen_decomp"
    assert result.neg_eigval_policy == "absolute"
    assert result.covariance_diagnostics.loc[(result.za[za], 33), "repaired"]
    assert result.uncertainty_eigenmodes["set_to_zero"].sum() == 0
    assert result.uncertainty_eigenmodes["absolute_value_applied"].sum() == 2
    assert not result.uncertainty_eigenmode_mts.empty
