"""Regression test based on the example1 notebook benchmark."""

from pathlib import Path

import numpy as np
import pytest
from uncertainties import unumpy as unp

DATA = Path(__file__).resolve().parent / "data" / "endfb_80"
ERRORR_DIR = DATA / "errorr"

EXPECTED = [
    ("keff", "total", "U-238", 31, 455, 455, 3.915966266547167e-10,
     1.5800338242176596e-11, 1.978880053602837e-05, 3.9922425347129545e-07),
    ("keff", "total", "U-238", 31, 455, 456, 0.0, 0.0, 0.0, 0.0),
    ("keff", "total", "U-238", 31, 456, 455, 0.0, 0.0, 0.0, 0.0),
    ("keff", "total", "U-238", 31, 456, 456, 1.854795884043281e-06,
     1.0789265661353025e-08, 0.001361908911801109, 3.9610819665921515e-06),
    ("keff", "total", "U-238", 33, 2, 2, 2.7141561299027834e-07,
     4.942906237962878e-08, 0.0005209756356973695, 4.7438938592074304e-05),
    ("keff", "total", "U-238", 33, 2, 18, 3.768940167079125e-08,
     3.9686847755246615e-09, 0.00019413758438486674, 1.0221320070762211e-05),
    ("keff", "total", "U-238", 33, 18, 2, 3.768940167079125e-08,
     3.9686847755246615e-09, 0.00019413758438486674, 1.0221320070762211e-05),
    ("keff", "total", "U-238", 33, 18, 18, 6.805892300699492e-07,
     6.609801787312911e-09, 0.0008249783209696781, 4.006045746477169e-06),
    ("keff", "total", "U-238", 34, 251, 251, 9.026015094404426e-08,
     2.850168070942206e-08, 0.0003004332720323171, 4.74342946715239e-05),
    ("keff", "total", "U-238", 35, 18, 18, 1.7940274977467052e-07,
     1.4840262245626608e-08, 0.00042355961773364386, 1.7518504626376974e-05),
]


def _value_parts(table, key):
    value = table.loc[key, "value"]
    return float(unp.nominal_values(value)), float(unp.std_devs(value))


def test_example1_notebook_uncertainty_regression():
    """
    Preserve the example1 uncertainty values validated outside pyNDUS.
    """
    pytest.importorskip("sandy")
    pytest.importorskip("serpentTools")
    from pyNDUS import Covariance, Sandwich, Sensitivity

    sens = Sensitivity(DATA / "example_sens0.m")
    covar_zais = {
        za:
        Covariance(za,
                   database=False,
                   group_structure=sens.group_structure,
                   egridname="ECCO-33",
                   cwd=ERRORR_DIR)
        for za in [922380]
    }

    sand = Sandwich(
        sens,
        covmat=covar_zais,
        list_MTs=[2, 18, 251, 455, 456],
        list_MFs=[31, 33, 34, 35],
        list_resp=["keff"],
        sum_MFs=False,
        include_MF=True,
    )

    assert sand.MFs2MTs[922380] == {
        "errorr31": [455, 456],
        "errorr33": [2, 18],
        "errorr34": [251],
        "errorr35": [18],
    }

    assert len(sand.uncertainty_variance_table) == len(EXPECTED)

    for (resp, mat, za, mf, mt_row, mt_col, var_nom, var_sd, sqrt_nom,
         sqrt_sd) in EXPECTED:
        key = (resp, mat, za, mf, mt_row, mt_col)
        got_var_nom, got_var_sd = _value_parts(sand.uncertainty_variance_table,
                                               key)
        got_sqrt_nom, got_sqrt_sd = _value_parts(
            sand.uncertainty_signed_sqrt_table, key)

        np.testing.assert_allclose(got_var_nom,
                                   var_nom,
                                   rtol=1e-12,
                                   atol=1e-18)
        np.testing.assert_allclose(got_var_sd, var_sd, rtol=1e-12, atol=1e-18)
        np.testing.assert_allclose(got_sqrt_nom,
                                   sqrt_nom,
                                   rtol=1e-12,
                                   atol=1e-18)
        np.testing.assert_allclose(got_sqrt_sd,
                                   sqrt_sd,
                                   rtol=1e-12,
                                   atol=1e-18)
