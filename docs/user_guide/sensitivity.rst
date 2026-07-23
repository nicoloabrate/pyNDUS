Sensitivity profiles
====================

Single-file input
-----------------

``Sensitivity`` recognizes Serpent files ending in ``_sens0.m`` and ERANOS
files using the supported ``.eranos33`` or ``.eranos1968`` suffixes.

.. code-block:: python

   from pyNDUS import Sensitivity

   sens = Sensitivity("model_sens0.m")

The object stores responses, materials, nuclides, MT identifiers, group
boundaries, mean sensitivity profiles, and—when available—the relative standard
deviations reported by the source calculation.

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
``(response, material, ZA, MT)``. Available policies are:

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

   avg, rsd = sens.get(
       resp=["keff"],
       mat=["total"],
       za=["U-235"],
       MT=[18],
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
Distinct source objects are treated as statistically independent.

See :doc:`../tutorials/sensitivity_algebra` for an executable walkthrough.
