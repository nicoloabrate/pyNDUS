Sensitivity profiles
====================

Single-file input
-----------------

``Sensitivity`` recognizes Serpent files ending in ``_sens0.m``, ERANOS files
using the supported ``.eranos33`` or ``.eranos1968`` suffixes, and MCNP
sensitivity outputs ending in ``.mcnp``.

.. code-block:: python

   from pyNDUS import Sensitivity

   sens = Sensitivity("model_sens0.m")

The object stores responses, materials, nuclides, sensitivity channels, group
boundaries, mean sensitivity profiles, and—when available—the relative standard
deviations reported by the source calculation.

Serpent may provide an aggregate sensitivity profile with ``ZAID=0``. pyNDUS
retains this profile for cross-checks and exposes its isotope label as
``"total"``. It is not an isotope covariance channel: uncertainty and
representativity calculations include it only when a covariance object is
explicitly available under key ``0``.

ERANOS perturbation columns are normalized through the same channel registry.
The supported labels are ``CAPTURE``, ``FISSION``, ``ELASTIC``,
``INELASTIC``, ``N,XN``, and ``NU``; ``N,XN`` is mapped to MT=16.
For ERANOS, ``CAPTURE`` is mapped to MT=102 except for ``B-10``, where
``get(..., za="B-10", channel="capture")`` resolves to MT=107 because the
dominant capture channel is ``n,alpha``.

For MCNP files, CELL- and MAT-based spatial zones are kept distinct on the
existing material axis. A zone such as ``spatial zone 1 covering cell(s): 149``
is exposed with a label like ``"profile 1 zone 1 cell 149"``, while
``spatial zone 1 covering material(s): 1`` becomes
``"profile 2 zone 1 material 1"``. If MCNP prints no explicit spatial-zone
block, pyNDUS falls back to ``"profile N"``. The structured metadata are also
available in ``sens.spatial_zones``.

MCNP channel names such as ``elastic``, ``n,gamma``, ``prompt chi``,
``delayed chi``, and ``elastic law`` are normalized to ``SensitivityChannel``
objects. Repeated isotope/channel blocks inside the same parsed spatial zone
are accumulated and their relative uncertainties are combined through absolute
uncertainties in quadrature.

MCNP ACE suffixes are preserved as metadata. If the same isotope is reported
with multiple ACE tables, for example ``92235.00c`` and ``92235.02c``, pyNDUS
stores the per-suffix profiles separately. By default, ``get(za=922350)``
raises an error when more than one suffix is available, because pyNDUS cannot
know whether the suffixes should be combined. Use ``temperature`` or
``ace_suffix`` to extract one MCNP table:

.. code-block:: python

   sens = Sensitivity("case.mcnp", mcnp_ace_temperatures={
       ".00c": 293.6,
       ".02c": 600.0,
   })

   avg_300K, rsd_300K = sens.get(za=922350, temperature=293.6)
   avg_00c, rsd_00c = sens.get(za=922350, ace_suffix=".00c")

If the printed MCNP coefficients are normalized to the same global response
and the intended perturbation is the same nuclear-data variation applied
coherently to all ACE suffixes, users can opt in to the summed profile:

.. code-block:: python

   sens = Sensitivity("case.mcnp", mcnp_ace_aggregation="sum")
   avg_sum, rsd_sum = sens.get(za=922350)

This option is deliberately explicit: if the reported quantities are ratios
with different denominators, the correct aggregate would require the
corresponding weights/denominators and is not inferred from the sensitivity
block alone.

``sens.ace_suffixes`` and ``sens.nuclide_instances`` record which suffixes
were present. The suffix-to-temperature mapping is optional and must be
provided by the user when needed; without it, ``temperature=...`` cannot be
resolved and users should select by ``ace_suffix`` instead.

Serpent perturbations and ENDF channels
---------------------------------------

Serpent reports some perturbations with descriptive names instead of plain
ENDF MT numbers. For example, ``chi prompt`` and ``ela leg mom 1`` are not
cross-section perturbations even though they are related to reactions that also
use familiar MT numbers.

For this reason, ``Sensitivity.channels`` is the canonical description of the
sensitivity-profile axis. Each key is a ``SensitivityChannel`` carrying the
average-side ENDF identifiers, the covariance-side ENDF identifiers, and
optionally the Legendre order ``L``. ``Sensitivity.MTs`` is retained only as a
compatibility alias for the same channel mapping.

.. code-block:: python

   sens = Sensitivity("case_sens0.m")

   print(sens.channels)
   print(sens.get_covariance_sensitivity_keys(35, 18))

Profiles can be selected directly by channel. The older ``MT=`` shortcut is
still accepted when it identifies exactly one channel; if the same MT appears
in more than one MF, pyNDUS raises an explicit ambiguity error.

.. code-block:: python

   from pyNDUS import SensitivityChannel

   chi_prompt = SensitivityChannel.from_alias("chi prompt")
   avg, rsd = sens.get(channel=chi_prompt)

   fission_xs = SensitivityChannel.from_endf(average_MF=3, average_MT=18)
   avg, rsd = sens.get(channel=fission_xs)

``SensitivityChannel.from_alias("chi prompt")`` is intended for reader labels
such as Serpent perturbation names,or  MCNP labels. Known aliases
are normalized to their ENDF-aware channel, so ``"scattering law"`` (MCNP) and
``"ela leg mom 1"`` (Serpent) both identify the first elastic Legendre moment.

``SensitivityChannel.from_endf(...)`` is intended for code that already knows
the ENDF identifiers. It accepts either average-side identifiers, such as
``average_MF=3, average_MT=18``, or covariance-side identifiers, such as
``covariance_MF=35, covariance_MT=18``. The latter returns ``chi prompt``.
For covariance channels that are not unique, pyNDUS asks for the missing
specifier instead of guessing; for example MF34/MT2 requires ``L=1`` or
``L=2``.

The main Serpent mappings are:

.. list-table::
   :header-rows: 1

   * - Serpent key
     - ENDF quantity
     - Covariance channel
   * - ``nubar total``
     - MF=1, MT=452
     - MF=31, MT=452
   * - ``nubar prompt``
     - MF=1, MT=456
     - MF=31, MT=456
   * - ``nubar delayed``
     - MF=1, MT=455
     - MF=31, MT=455
   * - ``chi prompt``
     - MF=5, MT=18
     - MF=35, MT=18
   * - ``chi delayed``
     - MF=5, MT=455
     - MF=35, MT=455
   * - ``ela leg mom 1``
     - MF=4, MT=2, L=1
     - MF=34, MT=251 or MF=34, MT=2, L=1
   * - ``ela leg mom 2``
     - MF=4, MT=2, L=2
     - not propagated by the current MF34 implementation

``chi total`` is a derived quantity and has no single ENDF covariance MT.
Numeric Serpent perturbations ``452``, ``455``, and ``456`` are treated as
MF=1 nubar quantities with MF=31 covariances, not as MF=3 cross sections.

Multi-file Serpent input
------------------------

Several Serpent sensitivity files can be merged:

.. code-block:: python

   sens = Sensitivity(
       ["part_1_sens0.m", "part_2_sens0.m"],
       duplicate_policy="raise",
   )

Before merging, the class verifies that all files use the same reader and the
same energy-group structure. The resulting object retains the same internal
array organization and public ``get`` interface as a single-file object.

Duplicate profiles are identified by the tuple
``(response, material, ZA, channel)``. Available policies are:

``raise``
   Stop and report the duplicate profile.

``keep_first``
   Preserve the first profile encountered and ignore later duplicates.

``keep_last``
   Replace the previously stored profile with the last profile encountered.

Extraction and ordering
-----------------------

The ``get`` method supports filtering by response, material, ZA/ZAIS, MT, and
energy group. Profiles are stored and returned in ascending energy order by
default, consistently with ``group_structure`` and covariance matrices. The
``group_order`` argument can still be set to ``"descending"`` when a high-to-low
view is useful.

.. code-block:: python

   fission_xs = SensitivityChannel.from_endf(average_MF=3, average_MT=18)

   avg, rsd = sens.get(
       resp=["keff"],
       mat=["total"],
       za=["U-235"],
       channel=fission_xs,
       group_order="ascending",
   )

Sensitivity algebra
-------------------

Sensitivity objects support algebra without modifying their input objects.
Scalar multiplication and division scale the sensitivity coefficients.
Because the coefficients are logarithmic derivatives, powers scale them by
the exponent, while multiplication and division of two underlying responses
correspond to addition and subtraction of their sensitivities:

.. code-block:: python

   half = sens / 2
   assert np.allclose((half + half).sens, sens.sens)

   unchanged = sens**1 + other**0
   product_sensitivity = sens * other
   ratio_sensitivity = sens / other

Scalar factors may also be ``uncertainties`` variables. In that case pyNDUS
uses the nominal value to scale the sensitivity coefficients and propagates
the uncertainty of the scalar into ``sens_rsd``. This is useful for EGPT-style
expressions where a sensitivity profile is normalized by a Monte Carlo
response such as ``keff``:

.. code-block:: python

   from uncertainties import ufloat

   keff = ufloat(1.00032, 0.00012)
   normalized = sens / keff

Binary operations require compatible energy-group boundaries. Metadata on the
response, material, ZAID and MT axes are handled according to one of three
policies:

``raise``
   Default. Require the same metadata sets in both objects; their order may
   differ because profiles are aligned by metadata value.

``intersect``
   Keep only metadata values common to both objects.

``zero``
   Keep the union of metadata values. Any profile absent from one object is
   treated as deterministic zero (average and standard deviation both zero).
   This is useful when sensitivities cover different nuclides or reactions:

   .. code-block:: python

      total = sens.combine(other, policy="zero")
      # Equivalent syntax for subsequent operators:
      total = sens.with_algebra_policy("zero") + other

The stored ``sens_rsd`` values are propagated through absolute standard
deviations. Expressions derived from the same source retain their correlation,
so ``sens / 2 + sens / 2`` reconstructs both the original averages and RSDs.
The same is true for repeated use of the same ``uncertainties`` variable.
Distinct source objects and distinct uncertainty variables are treated as
statistically independent.

See :doc:`../tutorials/sensitivity_algebra` for an executable walkthrough.
