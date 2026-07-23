"""Tests for Sandwich uncertainty propagation and missing covariance policies."""

import pytest
import numpy as np
import numpy.testing as npt

import pyNDUS.sandwich as sandwich_module
from pyNDUS.sandwich import Sandwich


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


def test_uncertainty_signed_sqrt_and_total_standard_deviation(monkeypatch):
    """Expose signed roots per term and root the sum for the total deviation."""
    monkeypatch.setattr(sandwich_module, "Sensitivity",
                        FakeSensitivityForInit)
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
        got = sand.uncertainty.loc[
            ("keff", "fuel", label, 33, mt_row), mt_col]
        npt.assert_allclose(got, expected, rtol=1e-14, atol=0.0)

    variance_12 = sand.uncertainty_variance.loc[
        ("keff", "fuel", label, 33, mt1), mt2]
    npt.assert_allclose(variance_12, -4.0, rtol=1e-14, atol=0.0)

    summary = sand.uncertainty_standard_deviation.loc[("keff", "fuel")]
    npt.assert_allclose(summary["variance"], 5.0, rtol=1e-14, atol=0.0)
    npt.assert_allclose(
        summary["standard_deviation"], np.sqrt(5.0),
        rtol=1e-14, atol=0.0)


def test_uncertainty_output_rejects_unknown_representation(monkeypatch):
    """Reject uncertainty output representations that are not documented."""
    monkeypatch.setattr(sandwich_module, "Sensitivity",
                        FakeSensitivityForInit)
    monkeypatch.setattr(sandwich_module, "Covariance", FakeCovZAForInit)

    za = 922350
    mt = 18
    mf = "errorr33"
    sens = FakeSensitivityForInit(
        {("keff", "fuel", za, mt): np.array([1.0, 0.0])},
        n_groups=2,
    )
    covmat = {
        za:
        FakeCovZAForInit({
            mf: {(mt, mt): np.eye(2)}
        })
    }

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
