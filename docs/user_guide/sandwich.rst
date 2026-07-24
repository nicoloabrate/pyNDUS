Uncertainty, similarity, and representativity
==============================================

The ``Sandwich`` class supports three calculation modes:

* uncertainty, selected by default;
* similarity, selected with ``similarity=True``;
* representativity, selected with ``representativity=True``.

Uncertainty calculations
------------------------

An uncertainty calculation combines sensitivity vectors with compatible
covariance blocks. Diagonal reaction terms and off-diagonal cross-reaction
terms are retained in the result structure.

By default, ``uncertainty`` contains variance contributions

.. math::

   q_{ij} = \mathbf{S}_i^{\mathrm{T}}\mathbf{C}_{ij}\mathbf{S}_j.

Set ``uncertainty_output="signed_sqrt"`` to display every matrix contribution
as

.. math::

   \widetilde{q}_{ij} =
   \operatorname{sign}(q_{ij})\sqrt{|q_{ij}|}.

This convention preserves the sign of negative covariance contributions, but
the transformed matrix elements are not additive standard deviations. The
physical total standard deviation is calculated by first summing the original
variance contributions and then taking the square root:

.. math::

   \sigma_R = \sqrt{\sum_{ij}q_{ij}}.

The raw matrix and its signed-square-root representation are always available
as ``uncertainty_variance`` and ``uncertainty_signed_sqrt``. The
``uncertainty_standard_deviation`` table reports the total variance and total
standard deviation for each response/material pair. ``uncertainty`` points to
the representation selected by ``uncertainty_output``. For example:

.. code-block:: python

   result = Sandwich(
       sensitivities,
       covmat=covariances,
       uncertainty_output="signed_sqrt",
   )
   print(result.uncertainty)
   print(result.uncertainty_standard_deviation)

A negative total variance has no real standard deviation and is reported as
``NaN``. Individual negative cross terms are expected and remain visible
through their negative signed square roots.

Similarity calculations
-----------------------

Similarity compares two sensitivity objects without covariance weighting. Both
objects must contain the requested responses and compatible energy groups.
Every matrix element compares one MT sensitivity vector from the first object
with one MT sensitivity vector from the second object. Its normalization uses
exactly those same two vectors:

.. math::

   E_{ij} =
   \frac{\mathbf{S}_{1,i}^{\mathrm{T}}\mathbf{S}_{2,j}}
   {\sqrt{\mathbf{S}_{1,i}^{\mathrm{T}}\mathbf{S}_{1,i}}
    \sqrt{\mathbf{S}_{2,j}^{\mathrm{T}}\mathbf{S}_{2,j}}}.

No norm accumulated over other MT reactions is used. Thus every coefficient
with non-zero vector norms lies in :math:`[-1,1]`; when an object is compared
with itself, each non-zero diagonal entry is one. Comparisons involving a
missing or zero-norm profile are reported as zero.

Representativity calculations
------------------------------

Representativity requires two sensitivity objects and a covariance dictionary.
The same covariance treatment must be applied consistently to the numerator and
the two normalization terms.

``representativity`` is a matrix decomposition of the normalized numerator into
ZA/MF/MT-pair contributions. Its elements are not independent representativity
coefficients. The physical total coefficient is exposed directly as
``representativity_total``, with one value per response:

.. math::

   r_{\mathrm{total}} =
   \sum_{z,i,j}
   \frac{\mathbf{S}_{A,z,i}^{\mathrm{T}}
         \mathbf{C}_{z,ij}\mathbf{S}_{B,z,j}}
        {\sqrt{\mathbf{S}_A^{\mathrm{T}}\mathbf{C}\mathbf{S}_A}
         \sqrt{\mathbf{S}_B^{\mathrm{T}}\mathbf{C}\mathbf{S}_B}}.

The numerator receives non-zero contributions only from profiles present in
both systems. With ``list_za=None``, each denominator nevertheless uses the
full isotope set of its own system. Thus isotopes present exclusively in A or
B contribute to that system's total uncertainty, but not to the numerator.

.. code-block:: python

   result = Sandwich(
       system_a,
       sens2=system_b,
       covmat=covariances,
       representativity=True,
   )
   print(result.representativity_total)

The detailed decomposition remains available through ``representativity`` and
``representativity_table``.

Missing-covariance policies
---------------------------

The numerical methods expose policies for covariance blocks that are not
available:

``zero``
   Preserve the legacy behavior and assign a zero contribution. For
   representativity, a fully missing covariance may make the normalized result
   undefined because both numerator and denominator vanish.

``raise``
   Stop the calculation with an explicit error identifying the missing ZA, MF,
   and MT information.

``assume``
   Construct a fallback relative covariance matrix. For a diagonal MT block,
   the default model is

   .. math::

      \mathbf{C}_{\mathrm{fallback}} = u^2\mathbf{I},

   where :math:`u` is the assumed relative standard deviation. Off-diagonal
   fallback blocks can be scaled by a user-supplied correlation coefficient.
   The default zero correlation avoids inventing cross-reaction correlations.

The assumed covariance is a modeling choice and should be reported explicitly
in any analysis that uses it.
