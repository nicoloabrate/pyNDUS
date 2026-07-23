"""Tests for Sandwich similarity and representativity calculations."""

import pytest
import numpy as np
import numpy.testing as npt

from pyNDUS.sandwich import Sandwich


class FakeSensitivity:
    """
    Minimal object supporting Sandwich.compute_similarity / compute_representativity:

    Required attributes:
      - reader ('serpent' -> expects mat='total')
      - responses (list)
      - materials (dict)
      - MTs (dict)
      - zaid (dict)
      - zais (dict)   (needed by compute_representativity)
      - n_groups (int)

    Required method:
      - get(resp=[...], mat=[...], MT=[...], za=[...], group_order="ascending")
        OR positional list style get([resp],[mat],[mt],[za], ...)
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

        # representativity builds za_dict_2 = dict(zip(sens2.zaid.keys(), sens2.zais.keys()))
        # so provide a zais dict with same ZA keys:
        self.zais = {za: str(za) for za in self.zaid.keys()}

        self.sens_rsd = None  # present in real Sensitivity objects

    def get(self, *args, **kwargs):
        """Return the sensitivity vector addressed by pyNDUS-style arguments."""
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
    Minimal covariance object for compute_representativity:
      - rcov has MF keys ('errorr31', ...)
      - get((mt1,mt2), MF=..., to_numpy=True) returns matrix
      - mat attribute exists (can be placeholder)
    """

    def __init__(self, mats_by_mf):
        """Store covariance blocks grouped by MF and MT pair."""
        self._mats_by_mf = mats_by_mf
        self.rcov = {mf: True for mf in mats_by_mf.keys()}
        self.mat = None

    def get(self, mt_pair, *, MF, to_numpy=False):
        """Return the covariance block for one MF and MT pair."""
        return self._mats_by_mf[MF][mt_pair]


def _get_df_value(df_matrix, resp, za_label, mt_row, mt_col):
    """Read one similarity or representativity value from the result matrix."""
    # similarity/representativity df_matrix index: (RESPONSE, ZA, MT_row)
    return df_matrix.loc[(resp, za_label, mt_row), mt_col]


def test_similarity_one_single_mt():
    """Return unit similarity for identical sensitivity vectors."""
    za = 922350
    za_dict = {za: "U235"}
    list_resp = ["keff"]
    list_MTs = {za: [18]}

    S = np.array([0.5, -0.2])

    sens = FakeSensitivity({("keff", "total", za, 18): S},
                           n_groups=2,
                           reader="serpent")
    sens2 = FakeSensitivity({("keff", "total", za, 18): S},
                            n_groups=2,
                            reader="serpent")

    sim_df, _ = Sandwich.compute_similarity(
        sens=sens,
        sens2=sens2,
        list_resp=list_resp,
        list_MTs=list_MTs,
        za_dict=za_dict,
        sens_MC=False,
        sigma=1,
    )

    got = _get_df_value(sim_df, "keff", "U235", 18, 18)
    npt.assert_allclose(got, 1.0, rtol=1e-14, atol=0.0)


def test_similarity_multi_mt_uses_pairwise_norms():
    """Normalize every MT pair with the two vectors in its numerator."""
    za = 942390
    za_dict = {za: "Pu239"}
    list_resp = ["keff"]
    list_MTs = {za: [18, 102]}

    S18 = np.array([3.0, 4.0])
    S102 = np.array([-4.0, 0.0])
    sens = FakeSensitivity(
        {
            ("keff", "total", za, 18): S18,
            ("keff", "total", za, 102): S102,
        },
        n_groups=2,
        reader="serpent",
    )

    sim_df, _ = Sandwich.compute_similarity(
        sens=sens,
        sens2=sens,
        list_resp=list_resp,
        list_MTs=list_MTs,
        za_dict=za_dict,
        sens_MC=False,
        sigma=1,
    )

    got = np.array(
        [
            [
                _get_df_value(sim_df, "keff", "Pu239", mt_row, mt_col)
                for mt_col in (18, 102)
            ]
            for mt_row in (18, 102)
        ]
    )
    expected_cross = S18.dot(S102) / (
        np.linalg.norm(S18) * np.linalg.norm(S102)
    )

    npt.assert_allclose(np.diag(got), np.ones(2), rtol=1e-14, atol=0.0)
    npt.assert_allclose(got[0, 1], expected_cross, rtol=1e-14, atol=0.0)
    npt.assert_allclose(got[1, 0], expected_cross, rtol=1e-14, atol=0.0)
    assert np.all(got >= -1.0)
    assert np.all(got <= 1.0)


def test_similarity_matches_normalised_dot_product():
    """Match the analytical normalized dot product for one MT."""
    za = 922350
    za_dict = {za: "U235"}
    list_resp = ["keff"]
    list_MTs = {za: [18]}

    S1 = np.array([0.5, -0.2])
    S2 = np.array([0.4, -0.1])

    sens = FakeSensitivity({("keff", "total", za, 18): S1},
                           n_groups=2,
                           reader="serpent")
    sens2 = FakeSensitivity({("keff", "total", za, 18): S2},
                            n_groups=2,
                            reader="serpent")

    sim_df, _ = Sandwich.compute_similarity(
        sens=sens,
        sens2=sens2,
        list_resp=list_resp,
        list_MTs=list_MTs,
        za_dict=za_dict,
        sens_MC=False,
        sigma=1,
    )

    expected = float(S1.dot(S2) / (np.sqrt(S1.dot(S1)) * np.sqrt(S2.dot(S2))))
    got = _get_df_value(sim_df, "keff", "U235", 18, 18)
    npt.assert_allclose(got, expected, rtol=1e-14, atol=0.0)


def test_similarity_symmetry():
    """Return the same similarity when the two systems are swapped."""
    za = 922350
    za_dict = {za: "U235"}
    list_resp = ["keff"]
    list_MTs = {za: [18]}

    S1 = np.array([0.5, -0.2])
    S2 = np.array([0.4, -0.1])

    sens_a = FakeSensitivity({("keff", "total", za, 18): S1},
                             n_groups=2,
                             reader="serpent")
    sens_b = FakeSensitivity({("keff", "total", za, 18): S2},
                             n_groups=2,
                             reader="serpent")

    sim_ab, _ = Sandwich.compute_similarity(sens=sens_a,
                                            sens2=sens_b,
                                            list_resp=list_resp,
                                            list_MTs=list_MTs,
                                            za_dict=za_dict,
                                            sens_MC=False,
                                            sigma=1)
    sim_ba, _ = Sandwich.compute_similarity(sens=sens_b,
                                            sens2=sens_a,
                                            list_resp=list_resp,
                                            list_MTs=list_MTs,
                                            za_dict=za_dict,
                                            sens_MC=False,
                                            sigma=1)

    got_ab = _get_df_value(sim_ab, "keff", "U235", 18, 18)
    got_ba = _get_df_value(sim_ba, "keff", "U235", 18, 18)
    npt.assert_allclose(got_ab, got_ba, rtol=1e-14, atol=0.0)


def test_representativity_one_single_mt():
    """Return unit representativity for identical systems with covariance."""
    za = 922350
    za_dict = {za: "U235"}
    list_resp = ["keff"]

    mf = "errorr31"
    mt = 18
    map_MF2MT = {za: {mf: [mt]}}

    S = np.array([0.5, -0.2])
    C = np.diag([0.1, 0.2])

    sens = FakeSensitivity({("keff", "total", za, mt): S},
                           n_groups=2,
                           reader="serpent")
    sens2 = FakeSensitivity({("keff", "total", za, mt): S},
                            n_groups=2,
                            reader="serpent")

    covmat = {za: FakeCovZA({mf: {(mt, mt): C}})}

    repr_df, _ = Sandwich.compute_representativity(
        sens=sens,
        sens2=sens2,
        covmat=covmat,
        list_resp=list_resp,
        map_MF2MT=map_MF2MT,
        za_dict=za_dict,
        sens_MC=False,
        sigma=1,
    )

    got = _get_df_value(repr_df, "keff", "U235", mt, mt)
    npt.assert_allclose(got, 1.0, rtol=1e-14, atol=0.0)


def test_representativity_missing_covariance_zero_policy_returns_nan():
    """Return NaN representativity when zero missing covariance makes norm zero."""
    za = 922350
    za_dict = {za: "U235"}
    list_resp = ["keff"]

    mf = "errorr31"
    mt = 18
    map_MF2MT = {za: {mf: [mt]}}

    S = np.array([0.5, -0.2])

    sens = FakeSensitivity({("keff", "total", za, mt): S},
                           n_groups=2,
                           reader="serpent")
    sens2 = FakeSensitivity({("keff", "total", za, mt): S},
                            n_groups=2,
                            reader="serpent")

    covmat = {}  # missing covariance everywhere

    repr_df, _ = Sandwich.compute_representativity(
        sens=sens,
        sens2=sens2,
        covmat=covmat,
        list_resp=list_resp,
        map_MF2MT=map_MF2MT,
        za_dict=za_dict,
        sens_MC=False,
        sigma=1,
        missing_cov="zero",
        missing_cov_rsd=0.20,
    )

    got = _get_df_value(repr_df, "keff", "U235", mt, mt)
    assert np.isnan(got)


def test_representativity_missing_covariance_raise_policy_raises():
    """Raise immediately when missing covariance uses the raise policy."""
    za = 922350
    za_dict = {za: "U235"}
    list_resp = ["keff"]

    mf = "errorr31"
    mt = 18
    map_MF2MT = {za: {mf: [mt]}}

    S = np.array([0.5, -0.2])

    sens = FakeSensitivity({("keff", "total", za, mt): S},
                           n_groups=2,
                           reader="serpent")
    sens2 = FakeSensitivity({("keff", "total", za, mt): S},
                            n_groups=2,
                            reader="serpent")

    covmat = {}  # missing covariance

    with pytest.raises(ValueError, match="Missing covariance"):
        Sandwich.compute_representativity(
            sens=sens,
            sens2=sens2,
            covmat=covmat,
            list_resp=list_resp,
            map_MF2MT=map_MF2MT,
            za_dict=za_dict,
            sens_MC=False,
            sigma=1,
            missing_cov="raise",
            missing_cov_rsd=0.20,
        )


def test_representativity_missing_covariance_assume_policy_identical_systems():
    """Use fallback covariance to give unit representativity for identical systems."""
    za = 922350
    za_dict = {za: "U235"}
    list_resp = ["keff"]

    mf = "errorr31"
    mt = 18
    map_MF2MT = {za: {mf: [mt]}}

    S = np.array([0.5, -0.2])

    sens = FakeSensitivity({("keff", "total", za, mt): S},
                           n_groups=2,
                           reader="serpent")
    sens2 = FakeSensitivity({("keff", "total", za, mt): S},
                            n_groups=2,
                            reader="serpent")

    covmat = {}  # missing covariance everywhere

    repr_df, _ = Sandwich.compute_representativity(
        sens=sens,
        sens2=sens2,
        covmat=covmat,
        list_resp=list_resp,
        map_MF2MT=map_MF2MT,
        za_dict=za_dict,
        sens_MC=False,
        sigma=1,
        missing_cov="assume",
        missing_cov_rsd=0.20,
        missing_cov_corr=0.0,
    )

    got = _get_df_value(repr_df, "keff", "U235", mt, mt)
    npt.assert_allclose(got, 1.0, rtol=1e-14, atol=0.0)


def test_representativity_missing_covariance_assume_policy_matches_fallback_formula(
):
    """Match the fallback-covariance representativity formula for two systems."""
    za = 922350
    za_dict = {za: "U235"}
    list_resp = ["keff"]

    mf = "errorr31"
    mt = 18
    map_MF2MT = {za: {mf: [mt]}}

    S1 = np.array([0.5, -0.2])
    S2 = np.array([0.4, -0.1])

    sens = FakeSensitivity({("keff", "total", za, mt): S1},
                           n_groups=2,
                           reader="serpent")
    sens2 = FakeSensitivity({("keff", "total", za, mt): S2},
                            n_groups=2,
                            reader="serpent")

    covmat = {}  # missing covariance

    repr_df, _ = Sandwich.compute_representativity(
        sens=sens,
        sens2=sens2,
        covmat=covmat,
        list_resp=list_resp,
        map_MF2MT=map_MF2MT,
        za_dict=za_dict,
        sens_MC=False,
        sigma=1,
        missing_cov="assume",
        missing_cov_rsd=0.20,
        missing_cov_corr=0.0,
    )

    expected = float(S1.dot(S2) / (np.sqrt(S1.dot(S1)) * np.sqrt(S2.dot(S2))))

    got = _get_df_value(repr_df, "keff", "U235", mt, mt)
    npt.assert_allclose(got, expected, rtol=1e-14, atol=0.0)


def test_representativity_invalid_missing_cov_policy_raises():
    """Reject unsupported missing-covariance policies."""
    za = 922350
    za_dict = {za: "U235"}
    list_resp = ["keff"]

    mf = "errorr31"
    mt = 18
    map_MF2MT = {za: {mf: [mt]}}

    S = np.array([0.5, -0.2])

    sens = FakeSensitivity({("keff", "total", za, mt): S},
                           n_groups=2,
                           reader="serpent")
    sens2 = FakeSensitivity({("keff", "total", za, mt): S},
                            n_groups=2,
                            reader="serpent")

    covmat = {}

    with pytest.raises(ValueError, match="Invalid missing_cov policy"):
        Sandwich.compute_representativity(
            sens=sens,
            sens2=sens2,
            covmat=covmat,
            list_resp=list_resp,
            map_MF2MT=map_MF2MT,
            za_dict=za_dict,
            sens_MC=False,
            sigma=1,
            missing_cov="banana",
            missing_cov_rsd=0.20,
        )
