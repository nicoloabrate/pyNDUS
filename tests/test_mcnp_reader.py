"""Regression tests for the MCNP sensitivity reader."""

from pathlib import Path

import numpy as np
import pytest

from pyNDUS import (Sensitivity, SensitivityChannel,
                    SensitivityNuclideInstance, SensitivitySpatialZone)

EXAMPLE_ROOT = Path(
    "/Users/nicoloabrate/Library/CloudStorage/OneDrive-PolitecnicodiTorino/"
    "newcleo/V&V/SA_UQ_pyNDUS")


def _example(name):
    path = EXAMPLE_ROOT / name
    if not path.exists():
        pytest.skip(f"MCNP sensitivity example {path} not available")
    return path


@pytest.mark.parametrize(
    "label, mt",
    [
        ("total", 1),
        ("elastic", 2),
        ("inelastic", 4),
        ("n,2nd", 11),
        ("n,2n", 16),
        ("n,3n", 17),
        ("fission", 18),
        ("n,f", 19),
        ("(First-Chance Fission)", 19),
        ("n,2nf", 20),
        ("n,nalpha", 22),
        ("n,n3alpha", 23),
        ("n,2nalpha", 24),
        ("n,np", 28),
        ("n,n2alpha", 29),
        ("n,2n2alpha", 30),
        ("n,nd", 32),
        ("n,nt", 33),
        ("n,n3he", 34),
        ("n,nd2alpha", 35),
        ("n,nt2alpha", 36),
        ("n,4n", 37),
        ("n,3nf", 38),
        ("n,2np", 41),
        ("n,3np", 42),
        ("n,n2p", 44),
        ("n,npalpha", 45),
        ("n,gamma", 102),
        ("n,p", 103),
        ("n,d", 104),
        ("n,t", 105),
        ("n,3he", 106),
        ("n,alpha", 107),
    ],
)
def test_mcnp_reaction_aliases_resolve_to_endf_mt(label, mt):
    """Resolve MCNP reaction labels to ENDF-6 MT cross-section channels."""
    channel = SensitivityChannel.from_alias(label)

    assert channel.average_MF == 3
    assert channel.average_MT == mt
    assert channel.covariance_MF == 33
    assert channel.covariance_MT == mt


def test_mcnp_special_aliases_resolve_to_endf_channels():
    """Resolve non-MF33 MCNP aliases without losing their covariance side."""
    assert SensitivityChannel.from_alias("total nu").average_MT == 452
    assert SensitivityChannel.from_alias("prompt nu").average_MT == 456
    assert SensitivityChannel.from_alias("delayed nu").average_MT == 455
    assert SensitivityChannel.from_alias("fission chi").average_MT is None
    assert SensitivityChannel.from_alias("prompt chi").covariance_MF == 35
    assert SensitivityChannel.from_alias("delayed chi").covariance_MT == 455


def test_mcnp_aliases_do_not_rename_canonical_xs_channels():
    """Keep historical Serpent/ERANOS channel names despite MCNP aliases."""
    assert SensitivityChannel.from_alias("n,gamma").name == "capture"
    assert SensitivityChannel.from_alias("n,2n").name == "n,xn"
    assert SensitivityChannel.from_alias("capture").name == "capture"
    assert SensitivityChannel.from_alias("n,xn").name == "n,xn"


@pytest.mark.parametrize("label", ["mt 18 xs", "xs mt 18", "xs 18"])
def test_serpent_xs_labels_resolve_to_endf_mt(label):
    """Resolve the two Serpent cross-section label orderings."""
    channel = SensitivityChannel.from_alias(label)

    assert channel.average_MF == 3
    assert channel.average_MT == 18
    assert channel.covariance_MF == 33
    assert channel.covariance_MT == 18


def test_mcnp_reader_fullcore_profile():
    """Read a one-region MCNP sensitivity output."""
    sens = Sensitivity(_example("sens_out_fullcore.mcnp"))

    assert sens.reader == "mcnp"
    assert sens.responses == ("keff", )
    assert list(sens.materials) == ["profile 1"]
    assert isinstance(sens.spatial_zones["profile 1"], SensitivitySpatialZone)
    assert sens.spatial_zones["profile 1"].kind == "profile"
    assert sens.energy_unit == "MeV"
    assert sens.n_groups == 33
    assert sens.sens.shape == (1, 1, 12, 9, 33)
    assert sens.sens_rsd.shape == sens.sens.shape

    elastic = SensitivityChannel.from_alias("elastic")
    avg, rsd = sens.get(resp="keff",
                        mat="profile 1",
                        channel=elastic,
                        za=942390,
                        group_order="ascending")
    np.testing.assert_allclose(avg.ravel()[1], -3.1268e-07)
    np.testing.assert_allclose(rsd.ravel()[1], 0.6897)


def test_mcnp_reader_splits_law_legendre_moments(tmp_path):
    """Read each MCNP law Legendre moment as a distinct channel."""
    path = tmp_path / "legendre_law.mcnp"
    path.write_text("""
 nuclear data keff sensitivity coefficients
      sensitivity profile      1

         94239.00c elastic law

        incident energy range:   0.0000E+00  1.0000E+36 MeV

                   legendre p1

                outgoing
            energy range (MeV)         sensitivity   rel. unc.

        1.0000E-11  1.0000E-07          1.0000E-01      0.1000

                   legendre p2

                outgoing
            energy range (MeV)         sensitivity   rel. unc.

        1.0000E-11  1.0000E-07          2.0000E-01      0.2000

         94239.00c n,2nf

            energy range (MeV)         sensitivity   rel. unc.

        1.0000E-11  1.0000E-07          3.0000E-01      0.3000
        """)

    sens = Sensitivity(path)

    p1 = SensitivityChannel.from_alias("elastic law legendre p1")
    p2 = SensitivityChannel.from_alias("elastic law legendre p2")
    n2nf = SensitivityChannel.from_alias("n,2nf")

    assert list(sens.channels) == [p1, p2, n2nf]
    avg, rsd = sens.get(resp="keff",
                        mat="profile 1",
                        channel=[p1, p2, n2nf],
                        za=942390,
                        group_order="ascending")
    np.testing.assert_allclose(avg.ravel(), [0.1, 0.2, 0.3])
    np.testing.assert_allclose(rsd.ravel(), [0.1, 0.2, 0.3])


def test_mcnp_reader_collapses_bare_profiles(tmp_path):
    """Sum MCNP sensitivity profiles when no cell/material zone is printed."""
    path = tmp_path / "bare_profiles.mcnp"
    path.write_text("""
 nuclear data keff sensitivity coefficients

      sensitivity profile      1

         92235.00c elastic

            energy range (MeV)         sensitivity   rel. unc.

        1.0000E-11  1.0000E-07          1.0000E-01      0.1000

      sensitivity profile      2

         92235.00c elastic

            energy range (MeV)         sensitivity   rel. unc.

        1.0000E-11  1.0000E-07          3.0000E-01      0.2000
        """)

    sens = Sensitivity(path)

    assert list(sens.materials) == ["profile"]
    assert isinstance(sens.spatial_zones["profile"], SensitivitySpatialZone)
    assert sens.spatial_zones["profile"].label == "profile"
    assert sens.sens.shape == (1, 1, 1, 1, 1)

    elastic = SensitivityChannel.from_alias("elastic")
    avg, rsd = sens.get(resp="keff",
                        mat="profile",
                        channel=elastic,
                        za=922350,
                        group_order="ascending")
    np.testing.assert_allclose(avg.ravel(), [0.4])
    expected_abs_unc = np.sqrt((0.1 * 0.1)**2 + (0.3 * 0.2)**2)
    np.testing.assert_allclose(rsd.ravel(), [expected_abs_unc / 0.4])


def test_mcnp_reader_regional_profiles_and_special_channels():
    """Read a two-region MCNP output and resolve MCNP special channels."""
    sens = Sensitivity(_example("sens_out_regions.mcnp"))

    assert sens.reader == "mcnp"
    assert list(sens.materials) == [
        "profile 1 zone 1 cell 149",
        "profile 1 zone 2 cell 150",
        "profile 2 zone 1 material 1",
        "profile 2 zone 2 material 2",
    ]
    assert sens.spatial_zones["profile 1 zone 1 cell 149"].kind == "cell"
    assert sens.spatial_zones["profile 1 zone 1 cell 149"].entries == (149, )
    assert sens.spatial_zones["profile 2 zone 1 material 1"].kind == "material"
    assert sens.spatial_zones["profile 2 zone 1 material 1"].entries == (1, )
    assert sens.sens.shape == (1, 4, 7, 9, 33)

    elastic = SensitivityChannel.from_alias("elastic")
    avg, rsd = sens.get(resp="keff",
                        mat="profile 2 zone 1 material 1",
                        channel=elastic,
                        za=942390,
                        group_order="ascending")
    np.testing.assert_allclose(avg.ravel()[3], -3.4274e-07)
    np.testing.assert_allclose(rsd.ravel()[3], 0.8991)

    assert SensitivityChannel.from_alias("prompt chi") in sens.channels
    assert SensitivityChannel.from_alias("delayed chi") in sens.channels
    assert SensitivityChannel.from_alias("scattering law") in sens.channels


def test_mcnp_reader_explicitly_aggregates_ace_suffixes_and_keeps_metadata(
        tmp_path):
    """Aggregate equal ZAIDs from different ACE suffixes only when requested."""
    path = tmp_path / "temperatures.mcnp"
    path.write_text("""
        text before sensitivity output

 nuclear data keff sensitivity coefficients

      sensitivity profile      1

         92235.00c elastic

            energy range (MeV)         sensitivity   rel. unc.

        1.0000E-11  1.0000E-07          1.0000E-01      0.1000
        1.0000E-07  5.4000E-07          2.0000E-01      0.2000

         92235.02c elastic

            energy range (MeV)         sensitivity   rel. unc.

        1.0000E-11  1.0000E-07          3.0000E-01      0.3000
        1.0000E-07  5.4000E-07          4.0000E-01      0.4000
        """)

    sens = Sensitivity(path,
                       mcnp_ace_temperatures={
                           "00c": 293.6,
                           ".02c": 600.0
                       },
                       mcnp_ace_aggregation="sum")

    assert list(sens.zaid) == [922350]
    assert sens.ace_suffixes[922350] == (".00c", ".02c")
    assert isinstance(sens.nuclide_instances[(922350, ".00c")],
                      SensitivityNuclideInstance)
    assert sens.nuclide_instances[(922350, ".00c")].temperature == 293.6
    assert sens.nuclide_instances[(922350, ".02c")].temperature == 600.0

    elastic = SensitivityChannel.from_alias("elastic")
    avg, rsd = sens.get(resp="keff",
                        mat="profile 1",
                        channel=elastic,
                        za=922350,
                        group_order="ascending")
    np.testing.assert_allclose(avg.ravel(), [0.4, 0.6])

    abs_unc = np.sqrt((np.array([0.1, 0.2]) * [0.1, 0.2])**2 +
                      (np.array([0.3, 0.4]) * [0.3, 0.4])**2)
    np.testing.assert_allclose(rsd.ravel(), abs_unc / np.array([0.4, 0.6]))

    avg_00, rsd_00 = sens.get(resp="keff",
                              mat="profile 1",
                              channel=elastic,
                              za=922350,
                              ace_suffix=".00c",
                              group_order="ascending")
    avg_02, rsd_02 = sens.get(resp="keff",
                              mat="profile 1",
                              channel=elastic,
                              za=922350,
                              ace_suffix="02c",
                              group_order="ascending")
    avg_600, rsd_600 = sens.get(resp="keff",
                                mat="profile 1",
                                channel=elastic,
                                za=922350,
                                temperature=600.0,
                                group_order="ascending")
    np.testing.assert_allclose(avg_00.ravel(), [0.1, 0.2])
    np.testing.assert_allclose(rsd_00.ravel(), [0.1, 0.2])
    np.testing.assert_allclose(avg_02.ravel(), [0.3, 0.4])
    np.testing.assert_allclose(rsd_02.ravel(), [0.3, 0.4])
    np.testing.assert_allclose(avg_600.ravel(), [0.3, 0.4])
    np.testing.assert_allclose(rsd_600.ravel(), [0.3, 0.4])


def test_mcnp_reader_rejects_implicit_ace_suffix_aggregation(tmp_path):
    """Avoid silently summing MCNP ACE suffixes for the same ZAID."""
    path = tmp_path / "temperatures.mcnp"
    path.write_text("""
 nuclear data keff sensitivity coefficients
      sensitivity profile      1
         92235.00c elastic
            energy range (MeV)         sensitivity   rel. unc.
        1.0000E-11  1.0000E-07          1.0000E-01      0.1000
         92235.02c elastic
            energy range (MeV)         sensitivity   rel. unc.
        1.0000E-11  1.0000E-07          3.0000E-01      0.3000
        """)
    sens = Sensitivity(path)
    elastic = SensitivityChannel.from_alias("elastic")

    with pytest.raises(ValueError, match="multiple ACE suffixes"):
        sens.get(resp="keff", mat="profile 1", channel=elastic, za=922350)


def test_mcnp_reader_per_temperature_example_file():
    """Read the MCNP example with multiple ACE suffixes per isotope."""
    sens = Sensitivity(_example("sens_mcnp_per_temperature.txt"),
                       mcnp_ace_temperatures={
                           ".00c": 300,
                           ".02c": 900
                       })

    assert sens.reader == "mcnp"
    assert list(sens.zaid) == [922350, 922380]
    assert sens.ace_suffixes[922350] == (".00c", ".02c")
    assert sens.ace_suffixes[922380] == (".00c", ".02c")
    assert sens.sens.shape == (1, 1, 2, 9, 33)
    assert sens.sens_nuclide_instances.shape == (1, 1, 4, 9, 33)

    elastic = SensitivityChannel.from_alias("elastic")
    avg_00, _ = sens.get(resp="keff",
                         mat="profile 1 zone 1 cell 149",
                         channel=elastic,
                         za=922350,
                         ace_suffix=".00c",
                         group_order="ascending")
    avg_02, _ = sens.get(resp="keff",
                         mat="profile 1 zone 1 cell 149",
                         channel=elastic,
                         za=922350,
                         ace_suffix=".02c",
                         group_order="ascending")
    avg_900, _ = sens.get(resp="keff",
                          mat="profile 1 zone 1 cell 149",
                          channel=elastic,
                          za=922350,
                          temperature=900,
                          group_order="ascending")

    np.testing.assert_allclose(avg_00.ravel()[11], 0.0)
    np.testing.assert_allclose(avg_02.ravel()[11], -9.7730e-10)
    np.testing.assert_allclose(avg_900.ravel()[11], -9.7730e-10)
    with pytest.raises(ValueError, match="multiple ACE suffixes"):
        sens.get(resp="keff",
                 mat="profile 1 zone 1 cell 149",
                 channel=elastic,
                 za=922350,
                 group_order="ascending")


def test_mcnp_temperature_filter_requires_user_temperature_map(tmp_path):
    """Report a clear error when temperature metadata are not available."""
    path = tmp_path / "temperatures.mcnp"
    path.write_text("""
 nuclear data keff sensitivity coefficients
      sensitivity profile      1
         92235.00c elastic
            energy range (MeV)         sensitivity   rel. unc.
        1.0000E-11  1.0000E-07          1.0000E-01      0.1000
        """)
    sens = Sensitivity(path)
    elastic = SensitivityChannel.from_alias("elastic")

    with pytest.raises(ValueError, match="Available temperatures: none"):
        sens.get(resp="keff",
                 mat="profile 1",
                 channel=elastic,
                 za=922350,
                 temperature=300)
