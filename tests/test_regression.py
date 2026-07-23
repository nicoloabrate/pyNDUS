"""Regression tests for sandwich calculations across MF/MT/ZA combinations."""

import numpy as np
import numpy.testing as npt

from pyNDUS.sandwich import Sandwich


class FakeSensitivity:
    """
    Minimal Sensitivity stub compatible with Sandwich.compute_* methods
    in this repo (keyword + positional list calling style).
    """

    def __init__(self, data, n_groups=2, reader="serpent"):
        """Store sparse sensitivity vectors and expose pyNDUS-like metadata."""
        self._data = data
        self.n_groups = n_groups
        self.reader = reader

        self.responses = sorted({k[0] for k in data.keys()})
        self.materials = {k[1]: None for k in data.keys()}
        self.zaid = {k[2]: None for k in data.keys()}
        self.MTs = {k[3]: None for k in data.keys()}

        # used by compute_representativity to build za_dict_2
        self.zais = {za: str(za) for za in self.zaid.keys()}
        self.sens_rsd = None

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
    Minimal covariance stub compatible with Sandwich.compute_* methods.
    """

    def __init__(self, mats_by_mf):
        """Store covariance blocks grouped by MF and MT pair."""
        self._mats_by_mf = mats_by_mf
        self.rcov = {mf: True for mf in mats_by_mf.keys()}
        self.mat = None

    def get(self, mt_pair, *, MF, to_numpy=False):
        """Return the covariance block for one MF and MT pair."""
        return self._mats_by_mf[MF][mt_pair]


def _get_unc_df_value(df_matrix, resp, mat, za_label, mt_row, mt_col):
    """Read one uncertainty contribution from the wide result matrix."""
    # uncertainty df index: (RESPONSE, MATERIAL, ZA, MT_row)
    return df_matrix.loc[(resp, mat, za_label, mt_row), mt_col]


def _get_simrepr_df_value(df_matrix, resp, za_label, mt_row, mt_col):
    """Read one similarity or representativity value from the result matrix."""
    # similarity/representativity df index: (RESPONSE, ZA, MT_row)
    return df_matrix.loc[(resp, za_label, mt_row), mt_col]


def test_regression_reference_case_numerics():
    """
    Frozen reference case (2 groups, U-235, MF=31).
    If this test changes, it must be an intentional, documented change
    in the numerical definitions/implementation.
    """
    za = 922350
    za_label = "U235"
    za_dict = {za: za_label}

    # --- Sensitivities (Serpent path --> mat='total' in similarity/representativity)
    mt1, mt2 = 18, 102
    S1 = np.array([0.5, -0.2])
    S2 = np.array([0.4, -0.1])

    # uncertainty uses list_mat explicitly; keep 'fuel' to exercise that path
    sens_unc = FakeSensitivity(
        {
            ("keff", "fuel", za, mt1): S1,
            ("keff", "fuel", za, mt2): S2,
        },
        n_groups=2,
        reader="serpent",
    )

    # similarity/representativity force mat='total' for serpent in your implementation
    sens_ref = FakeSensitivity({("keff", "total", za, mt1): S1},
                               n_groups=2,
                               reader="serpent")
    sens_app = FakeSensitivity({("keff", "total", za, mt1): S2},
                               n_groups=2,
                               reader="serpent")

    # --- Covariance blocks (MF=31)
    mf = "errorr31"
    C11 = np.diag([0.1, 0.2])
    C22 = np.diag([0.3, 0.4])
    C12 = np.diag([0.01, 0.02])
    C21 = np.diag([0.01, 0.02])

    cov_unc = {
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

    cov_repr = {za: FakeCovZA({mf: {(mt1, mt1): C11}})}

    # --- 1) Uncertainty regression (matrix entries)
    map_MF2MT_unc = {za: {mf: [mt1, mt2]}}

    unc_df, _ = Sandwich.compute_uncertainty(
        sens=sens_unc,
        covmat=cov_unc,
        list_resp=["keff"],
        list_mat=["fuel"],
        map_MF2MT=map_MF2MT_unc,
        za_dict=za_dict,
        sens_MC=False,
        sigma=1,
    )

    # Frozen expected values (computed analytically)
    exp_11 = 0.033
    exp_22 = 0.052
    exp_12 = 0.0024
    exp_21 = 0.0024

    npt.assert_allclose(_get_unc_df_value(unc_df, "keff", "fuel", za_label,
                                          mt1, mt1),
                        exp_11,
                        rtol=1e-14,
                        atol=0.0)
    npt.assert_allclose(_get_unc_df_value(unc_df, "keff", "fuel", za_label,
                                          mt2, mt2),
                        exp_22,
                        rtol=1e-14,
                        atol=0.0)
    npt.assert_allclose(_get_unc_df_value(unc_df, "keff", "fuel", za_label,
                                          mt1, mt2),
                        exp_12,
                        rtol=1e-14,
                        atol=0.0)
    npt.assert_allclose(_get_unc_df_value(unc_df, "keff", "fuel", za_label,
                                          mt2, mt1),
                        exp_21,
                        rtol=1e-14,
                        atol=0.0)

    # --- 2) Similarity regression (single MT)
    sim_df, _ = Sandwich.compute_similarity(
        sens=sens_ref,
        sens2=sens_app,
        list_resp=["keff"],
        list_MTs={za: [mt1]},
        za_dict=za_dict,
        sens_MC=False,
        sigma=1,
    )

    exp_sim = 0.9908301680442989
    npt.assert_allclose(_get_simrepr_df_value(sim_df, "keff", za_label, mt1,
                                              mt1),
                        exp_sim,
                        rtol=1e-14,
                        atol=0.0)

    # --- 3) Representativity regression (single MT, covariance-weighted)
    map_MF2MT_repr = {za: {mf: [mt1]}}

    repr_df, _ = Sandwich.compute_representativity(
        sens=sens_ref,
        sens2=sens_app,
        covmat=cov_repr,
        list_resp=["keff"],
        map_MF2MT=map_MF2MT_repr,
        za_dict=za_dict,
        sens_MC=False,
        sigma=1,
    )

    exp_repr = 0.9847319278346619
    npt.assert_allclose(_get_simrepr_df_value(repr_df, "keff", za_label, mt1,
                                              mt1),
                        exp_repr,
                        rtol=1e-14,
                        atol=0.0)


def test_regression_multi_mf_single_mt_single_za():
    """Keep MF31 and MF33 diagonal contributions for one isotope distinct."""
    za = 922350
    za_label = "U235"
    za_dict = {za: za_label}

    mf33 = "errorr33"
    mf31 = "errorr31"

    mt33 = 18
    mt31 = 455

    S33 = np.array([0.5, -0.2])
    S31 = np.array([0.2, 0.1])

    sens = FakeSensitivity(
        {
            ("keff", "fuel", za, mt33): S33,
            ("keff", "fuel", za, mt31): S31,
        },
        n_groups=2,
        reader="serpent",
    )

    C33 = np.diag([0.10, 0.20])
    C31 = np.diag([0.05, 0.07])

    covmat = {
        za: FakeCovZA({
            mf33: {
                (mt33, mt33): C33
            },
            mf31: {
                (mt31, mt31): C31
            },
        })
    }

    map_MF2MT = {za: {mf33: [mt33], mf31: [mt31]}}

    unc_df, _ = Sandwich.compute_uncertainty(
        sens=sens,
        covmat=covmat,
        list_resp=["keff"],
        list_mat=["fuel"],
        map_MF2MT=map_MF2MT,
        za_dict=za_dict,
        sens_MC=False,
        sigma=1,
    )

    exp_33 = float(S33.T @ C33 @ S33)
    exp_31 = float(S31.T @ C31 @ S31)

    npt.assert_allclose(_get_unc_df_value(unc_df, "keff", "fuel", za_label,
                                          mt33, mt33),
                        exp_33,
                        rtol=1e-14,
                        atol=0.0)
    npt.assert_allclose(_get_unc_df_value(unc_df, "keff", "fuel", za_label,
                                          mt31, mt31),
                        exp_31,
                        rtol=1e-14,
                        atol=0.0)


def test_regression_multi_mf_multi_mt_single_za():
    """Preserve same-isotope cross terms separately within each MF section."""
    za = 922350
    za_label = "U235"
    za_dict = {za: za_label}

    mf33 = "errorr33"
    mf31 = "errorr31"

    mt1, mt2 = 18, 102  # MF33
    mt3, mt4 = 455, 456  # MF31

    S1 = np.array([0.5, -0.2])
    S2 = np.array([0.4, -0.1])
    S3 = np.array([0.2, 0.1])
    S4 = np.array([-0.1, 0.3])

    sens = FakeSensitivity(
        {
            ("keff", "fuel", za, mt1): S1,
            ("keff", "fuel", za, mt2): S2,
            ("keff", "fuel", za, mt3): S3,
            ("keff", "fuel", za, mt4): S4,
        },
        n_groups=2,
        reader="serpent",
    )

    # MF33 blocks
    C11_33 = np.diag([0.10, 0.20])
    C22_33 = np.diag([0.30, 0.40])
    C12_33 = np.diag([0.01, 0.02])
    C21_33 = np.diag([0.01, 0.02])

    # MF31 blocks
    C33_31 = np.diag([0.05, 0.07])
    C44_31 = np.diag([0.08, 0.09])
    C34_31 = np.diag([0.003, 0.004])
    C43_31 = np.diag([0.003, 0.004])

    covmat = {
        za:
        FakeCovZA({
            mf33: {
                (mt1, mt1): C11_33,
                (mt2, mt2): C22_33,
                (mt1, mt2): C12_33,
                (mt2, mt1): C21_33,
            },
            mf31: {
                (mt3, mt3): C33_31,
                (mt4, mt4): C44_31,
                (mt3, mt4): C34_31,
                (mt4, mt3): C43_31,
            },
        })
    }

    map_MF2MT = {za: {mf33: [mt1, mt2], mf31: [mt3, mt4]}}

    unc_df, _ = Sandwich.compute_uncertainty(
        sens=sens,
        covmat=covmat,
        list_resp=["keff"],
        list_mat=["fuel"],
        map_MF2MT=map_MF2MT,
        za_dict=za_dict,
        sens_MC=False,
        sigma=1,
    )

    # Expected MF33
    exp_11 = float(S1.T @ C11_33 @ S1)
    exp_22 = float(S2.T @ C22_33 @ S2)
    exp_12 = float(S1.T @ C12_33 @ S2)
    exp_21 = float(S2.T @ C21_33 @ S1)

    # Expected MF31
    exp_33 = float(S3.T @ C33_31 @ S3)
    exp_44 = float(S4.T @ C44_31 @ S4)
    exp_34 = float(S3.T @ C34_31 @ S4)
    exp_43 = float(S4.T @ C43_31 @ S3)

    npt.assert_allclose(_get_unc_df_value(unc_df, "keff", "fuel", za_label,
                                          mt1, mt1),
                        exp_11,
                        rtol=1e-14,
                        atol=0.0)
    npt.assert_allclose(_get_unc_df_value(unc_df, "keff", "fuel", za_label,
                                          mt2, mt2),
                        exp_22,
                        rtol=1e-14,
                        atol=0.0)
    npt.assert_allclose(_get_unc_df_value(unc_df, "keff", "fuel", za_label,
                                          mt1, mt2),
                        exp_12,
                        rtol=1e-14,
                        atol=0.0)
    npt.assert_allclose(_get_unc_df_value(unc_df, "keff", "fuel", za_label,
                                          mt2, mt1),
                        exp_21,
                        rtol=1e-14,
                        atol=0.0)

    npt.assert_allclose(_get_unc_df_value(unc_df, "keff", "fuel", za_label,
                                          mt3, mt3),
                        exp_33,
                        rtol=1e-14,
                        atol=0.0)
    npt.assert_allclose(_get_unc_df_value(unc_df, "keff", "fuel", za_label,
                                          mt4, mt4),
                        exp_44,
                        rtol=1e-14,
                        atol=0.0)
    npt.assert_allclose(_get_unc_df_value(unc_df, "keff", "fuel", za_label,
                                          mt3, mt4),
                        exp_34,
                        rtol=1e-14,
                        atol=0.0)
    npt.assert_allclose(_get_unc_df_value(unc_df, "keff", "fuel", za_label,
                                          mt4, mt3),
                        exp_43,
                        rtol=1e-14,
                        atol=0.0)


def test_regression_multi_mf_single_mt_multi_za():
    """Compute separate MF contributions for multiple isotopes."""
    za_u, za_pu = 922350, 942390
    za_dict = {za_u: "U235", za_pu: "Pu239"}

    mf33 = "errorr33"
    mf31 = "errorr31"
    mt33 = 18
    mt31 = 455

    Su33 = np.array([0.5, -0.2])
    Su31 = np.array([0.2, 0.1])
    Sp33 = np.array([0.1, 0.3])
    Sp31 = np.array([-0.2, 0.2])

    sens = FakeSensitivity(
        {
            ("keff", "fuel", za_u, mt33): Su33,
            ("keff", "fuel", za_u, mt31): Su31,
            ("keff", "fuel", za_pu, mt33): Sp33,
            ("keff", "fuel", za_pu, mt31): Sp31,
        },
        n_groups=2,
        reader="serpent",
    )

    C33_u = np.diag([0.10, 0.20])
    C31_u = np.diag([0.05, 0.07])
    C33_p = np.diag([0.07, 0.09])
    C31_p = np.diag([0.02, 0.03])

    covmat = {
        za_u:
        FakeCovZA({
            mf33: {
                (mt33, mt33): C33_u
            },
            mf31: {
                (mt31, mt31): C31_u
            }
        }),
        za_pu:
        FakeCovZA({
            mf33: {
                (mt33, mt33): C33_p
            },
            mf31: {
                (mt31, mt31): C31_p
            }
        }),
    }

    map_MF2MT = {
        za_u: {
            mf33: [mt33],
            mf31: [mt31]
        },
        za_pu: {
            mf33: [mt33],
            mf31: [mt31]
        },
    }

    unc_df, _ = Sandwich.compute_uncertainty(
        sens=sens,
        covmat=covmat,
        list_resp=["keff"],
        list_mat=["fuel"],
        map_MF2MT=map_MF2MT,
        za_dict=za_dict,
        sens_MC=False,
        sigma=1,
    )

    exp_u33 = float(Su33.T @ C33_u @ Su33)
    exp_u31 = float(Su31.T @ C31_u @ Su31)
    exp_p33 = float(Sp33.T @ C33_p @ Sp33)
    exp_p31 = float(Sp31.T @ C31_p @ Sp31)

    npt.assert_allclose(_get_unc_df_value(unc_df, "keff", "fuel", "U235", mt33,
                                          mt33),
                        exp_u33,
                        rtol=1e-14,
                        atol=0.0)
    npt.assert_allclose(_get_unc_df_value(unc_df, "keff", "fuel", "U235", mt31,
                                          mt31),
                        exp_u31,
                        rtol=1e-14,
                        atol=0.0)
    npt.assert_allclose(_get_unc_df_value(unc_df, "keff", "fuel", "Pu239",
                                          mt33, mt33),
                        exp_p33,
                        rtol=1e-14,
                        atol=0.0)
    npt.assert_allclose(_get_unc_df_value(unc_df, "keff", "fuel", "Pu239",
                                          mt31, mt31),
                        exp_p31,
                        rtol=1e-14,
                        atol=0.0)


def test_regression_multi_mf_multi_mt_multi_za():
    """Freeze multi-MF, multi-MT uncertainty values for multiple isotopes."""
    za_u, za_pu = 922350, 942390
    za_dict = {za_u: "U235", za_pu: "Pu239"}

    mf33 = "errorr33"
    mf31 = "errorr31"

    mt1, mt2 = 18, 102  # MF33
    mt3, mt4 = 455, 456  # MF31

    # U235 sensitivities
    Su1 = np.array([0.5, -0.2])
    Su2 = np.array([0.4, -0.1])
    Su3 = np.array([0.2, 0.1])
    Su4 = np.array([-0.1, 0.3])

    # Pu239 sensitivities
    Sp1 = np.array([0.1, 0.3])
    Sp2 = np.array([0.2, -0.2])
    Sp3 = np.array([-0.2, 0.2])
    Sp4 = np.array([0.3, 0.1])

    sens = FakeSensitivity(
        {
            ("keff", "fuel", za_u, mt1): Su1,
            ("keff", "fuel", za_u, mt2): Su2,
            ("keff", "fuel", za_u, mt3): Su3,
            ("keff", "fuel", za_u, mt4): Su4,
            ("keff", "fuel", za_pu, mt1): Sp1,
            ("keff", "fuel", za_pu, mt2): Sp2,
            ("keff", "fuel", za_pu, mt3): Sp3,
            ("keff", "fuel", za_pu, mt4): Sp4,
        },
        n_groups=2,
        reader="serpent",
    )

    # --- U235 covariance blocks
    C11u_33 = np.diag([0.10, 0.20])
    C22u_33 = np.diag([0.30, 0.40])
    C12u_33 = np.diag([0.01, 0.02])
    C21u_33 = np.diag([0.01, 0.02])

    C33u_31 = np.diag([0.05, 0.07])
    C44u_31 = np.diag([0.08, 0.09])
    C34u_31 = np.diag([0.003, 0.004])
    C43u_31 = np.diag([0.003, 0.004])

    # --- Pu239 covariance blocks (different)
    C11p_33 = np.diag([0.07, 0.09])
    C22p_33 = np.diag([0.11, 0.13])
    C12p_33 = np.diag([0.004, 0.006])
    C21p_33 = np.diag([0.004, 0.006])

    C33p_31 = np.diag([0.02, 0.03])
    C44p_31 = np.diag([0.05, 0.08])
    C34p_31 = np.diag([0.001, 0.002])
    C43p_31 = np.diag([0.001, 0.002])

    covmat = {
        za_u:
        FakeCovZA({
            mf33: {
                (mt1, mt1): C11u_33,
                (mt2, mt2): C22u_33,
                (mt1, mt2): C12u_33,
                (mt2, mt1): C21u_33
            },
            mf31: {
                (mt3, mt3): C33u_31,
                (mt4, mt4): C44u_31,
                (mt3, mt4): C34u_31,
                (mt4, mt3): C43u_31
            },
        }),
        za_pu:
        FakeCovZA({
            mf33: {
                (mt1, mt1): C11p_33,
                (mt2, mt2): C22p_33,
                (mt1, mt2): C12p_33,
                (mt2, mt1): C21p_33
            },
            mf31: {
                (mt3, mt3): C33p_31,
                (mt4, mt4): C44p_31,
                (mt3, mt4): C34p_31,
                (mt4, mt3): C43p_31
            },
        }),
    }

    map_MF2MT = {
        za_u: {
            mf33: [mt1, mt2],
            mf31: [mt3, mt4]
        },
        za_pu: {
            mf33: [mt1, mt2],
            mf31: [mt3, mt4]
        },
    }

    unc_df, _ = Sandwich.compute_uncertainty(
        sens=sens,
        covmat=covmat,
        list_resp=["keff"],
        list_mat=["fuel"],
        map_MF2MT=map_MF2MT,
        za_dict=za_dict,
        sens_MC=False,
        sigma=1,
    )

    # U235 expected
    exp_u11 = float(Su1.T @ C11u_33 @ Su1)
    exp_u22 = float(Su2.T @ C22u_33 @ Su2)
    exp_u12 = float(Su1.T @ C12u_33 @ Su2)
    exp_u21 = float(Su2.T @ C21u_33 @ Su1)
    exp_u33 = float(Su3.T @ C33u_31 @ Su3)
    exp_u44 = float(Su4.T @ C44u_31 @ Su4)
    exp_u34 = float(Su3.T @ C34u_31 @ Su4)
    exp_u43 = float(Su4.T @ C43u_31 @ Su3)

    # Pu239 expected
    exp_p11 = float(Sp1.T @ C11p_33 @ Sp1)
    exp_p22 = float(Sp2.T @ C22p_33 @ Sp2)
    exp_p12 = float(Sp1.T @ C12p_33 @ Sp2)
    exp_p21 = float(Sp2.T @ C21p_33 @ Sp1)
    exp_p33 = float(Sp3.T @ C33p_31 @ Sp3)
    exp_p44 = float(Sp4.T @ C44p_31 @ Sp4)
    exp_p34 = float(Sp3.T @ C34p_31 @ Sp4)
    exp_p43 = float(Sp4.T @ C43p_31 @ Sp3)

    # Assert U235
    npt.assert_allclose(_get_unc_df_value(unc_df, "keff", "fuel", "U235", mt1,
                                          mt1),
                        exp_u11,
                        rtol=1e-14,
                        atol=0.0)
    npt.assert_allclose(_get_unc_df_value(unc_df, "keff", "fuel", "U235", mt2,
                                          mt2),
                        exp_u22,
                        rtol=1e-14,
                        atol=0.0)
    npt.assert_allclose(_get_unc_df_value(unc_df, "keff", "fuel", "U235", mt1,
                                          mt2),
                        exp_u12,
                        rtol=1e-14,
                        atol=0.0)
    npt.assert_allclose(_get_unc_df_value(unc_df, "keff", "fuel", "U235", mt2,
                                          mt1),
                        exp_u21,
                        rtol=1e-14,
                        atol=0.0)
    npt.assert_allclose(_get_unc_df_value(unc_df, "keff", "fuel", "U235", mt3,
                                          mt3),
                        exp_u33,
                        rtol=1e-14,
                        atol=0.0)
    npt.assert_allclose(_get_unc_df_value(unc_df, "keff", "fuel", "U235", mt4,
                                          mt4),
                        exp_u44,
                        rtol=1e-14,
                        atol=0.0)
    npt.assert_allclose(_get_unc_df_value(unc_df, "keff", "fuel", "U235", mt3,
                                          mt4),
                        exp_u34,
                        rtol=1e-14,
                        atol=0.0)
    npt.assert_allclose(_get_unc_df_value(unc_df, "keff", "fuel", "U235", mt4,
                                          mt3),
                        exp_u43,
                        rtol=1e-14,
                        atol=0.0)

    # Assert Pu239
    npt.assert_allclose(_get_unc_df_value(unc_df, "keff", "fuel", "Pu239", mt1,
                                          mt1),
                        exp_p11,
                        rtol=1e-14,
                        atol=0.0)
    npt.assert_allclose(_get_unc_df_value(unc_df, "keff", "fuel", "Pu239", mt2,
                                          mt2),
                        exp_p22,
                        rtol=1e-14,
                        atol=0.0)
    npt.assert_allclose(_get_unc_df_value(unc_df, "keff", "fuel", "Pu239", mt1,
                                          mt2),
                        exp_p12,
                        rtol=1e-14,
                        atol=0.0)
    npt.assert_allclose(_get_unc_df_value(unc_df, "keff", "fuel", "Pu239", mt2,
                                          mt1),
                        exp_p21,
                        rtol=1e-14,
                        atol=0.0)
    npt.assert_allclose(_get_unc_df_value(unc_df, "keff", "fuel", "Pu239", mt3,
                                          mt3),
                        exp_p33,
                        rtol=1e-14,
                        atol=0.0)
    npt.assert_allclose(_get_unc_df_value(unc_df, "keff", "fuel", "Pu239", mt4,
                                          mt4),
                        exp_p44,
                        rtol=1e-14,
                        atol=0.0)
    npt.assert_allclose(_get_unc_df_value(unc_df, "keff", "fuel", "Pu239", mt3,
                                          mt4),
                        exp_p34,
                        rtol=1e-14,
                        atol=0.0)
    npt.assert_allclose(_get_unc_df_value(unc_df, "keff", "fuel", "Pu239", mt4,
                                          mt3),
                        exp_p43,
                        rtol=1e-14,
                        atol=0.0)
