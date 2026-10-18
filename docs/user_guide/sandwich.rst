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

When ``list_MTs=None``, ``Sandwich`` excludes ENDF MT=1 from the automatic
selection because it is the total cross section rather than an independent
reaction channel. Pass ``list_MTs=[1]`` (or include ``1`` in the list) when
its covariance contribution is intentionally required.

If the sensitivity reader provides Monte Carlo relative standard deviations
(``sens_rsd``), the sensitivity coefficients are treated as uncertain
quantities. pyNDUS converts them to ``uncertainties`` variables and propagates
that extra Monte Carlo component through the sandwich algebra. The ``sigma``
argument controls the assumed width of the MC error bar: the default is
``sigma=2``, i.e. two times the reported MC relative standard deviation. Use
``sigma=1`` for a one-sigma propagation. When no ``sens_rsd`` array is present,
``sigma`` is ignored.

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

Non-PSD covariance policies
---------------------------

A covariance matrix must be symmetric positive semidefinite (PSD), i.e. the
eigenvalue spectrum must be non-negative. For every ZA and MF, ``Sandwich``
assembles one matrix from exactly the MTs selected by ``list_MTs``:

.. math::

   \mathbf{C}_{z,f} = [\mathbf{C}_{mn}]_{m,n\in\mathcal{M}_{z,f}}.

The PSD diagnostic diagonalizes
:math:`(\mathbf{C}_{z,f}+\mathbf{C}_{z,f}^{\mathrm{T}})/2`. This
symmetrization removes numerical or input asymmetry for the eigendecomposition;
it does not by itself make an indefinite matrix PSD. An eigenvalue is treated
as significantly negative when

.. math::

   \lambda_k < -(a_{\mathrm{tol}} + r_{\mathrm{tol}}
   \max_j |\lambda_j|).

The constructor arguments ``non_psd_rtol`` and ``non_psd_atol`` control these
tolerances. ``non_psd_policy`` selects one of three behaviors:

``warn``
   Prints ``NonPositiveSemidefiniteWarning`` once for each affected ZA/MF block
   and continue with the unmodified selected covariance. This preserves the raw
   sandwich calculation, including a possible negative total variance or a
   representativity coefficient outside its usual PSD bounds. 
   This is suggested only as a temporary diagnostic for a known non-PSD covariance. 
   It is not recommended for production calculations because the resulting uncertainty 
   may be unphysical.

``raise``
   Stops before propagation with ``NonPositiveSemidefiniteError``. The error
   identifies the ZA, MF, selected MTs, minimum eigenvalue, and tolerance. Use
   this policy when an indefinite evaluated covariance must invalidate an
   analysis. This is the default policy, so accepting or repairing a non-PSD
   covariance always requires an explicit user choice.
   This is the default choice.

``eigen_decomp``
   Prints ``NonPositiveSemidefiniteWarning`` and propagate the reconstructed PSD
   matrix obtained by replacing its negative eigenvalues. The replacement is
   selected with ``neg_eigval_policy``. The default ``"zero"`` uses

   .. math::

      \mathbf{C}^{+}_{z,f} = \mathbf{Q}
      \operatorname{diag}(\max(\lambda_k,0))\mathbf{Q}^{\mathrm{T}}.

   The alternative ``"absolute"`` uses

   .. math::

      \mathbf{C}^{|\cdot|}_{z,f} = \mathbf{Q}
      \operatorname{diag}(|\lambda_k|)\mathbf{Q}^{\mathrm{T}}.

   Every negative eigenvalue, including a negative value within the numerical
   warning tolerance, is replaced. The reconstructed matrix is used consistently
   for all MT-pair contributions. In representativity it is also the same matrix
   used in the numerator and both denominator variances. Both choices produce a
   PSD matrix; ``"absolute"`` retains the magnitude of each negative mode as a
   positive contribution instead of removing it.

For example:

.. code-block:: python

   result = Sandwich(
       sensitivities,
       sens2=other_sensitivities,
       covmat=covariances,
       representativity=True,
       list_za=[922380],
       list_MFs=[33],
       list_MTs=[2, 18, 102],
       non_psd_policy="eigen_decomp",
       neg_eigval_policy="absolute",
   )
   print(result.covariance_diagnostics)

The eigendecomposition is performed after the ZA/MF/MT selection. Changing the
selected MTs changes the matrix being tested and can therefore change both its
spectrum and an ``eigen_decomp`` repair. Replacing negative eigenvalues by zero
or by their absolute values is an explicit regularization choice, not a
reconstruction of the unknown physically correct covariance, and neither choice
is guaranteed to bound that unknown uncertainty from above or below.

Eigenmode and per-MT uncertainty diagnostics
--------------------------------------------

Uncertainty calculations expose two modal tables. In
``uncertainty_eigenmodes``, the raw contribution of mode :math:`k` is

.. math::

   v_k = \lambda_k(\mathbf{q}_k^{\mathrm{T}}\mathbf{S})^2.

``raw_variance`` always uses the evaluated eigenvalue. ``used_variance`` uses
the propagated eigenvalue, which is zero for a negative mode with
``neg_eigval_policy="zero"`` and positive with
``neg_eigval_policy="absolute"``. The original negative value remains
available in ``raw_variance``. Modes follow the ascending order returned by
``numpy.linalg.eigh`` and ``EIGENMODE`` is zero-based. Consequently, summing
``used_variance`` over the modes and then over ZA/MF reproduces the propagated
total variance. The ``set_to_zero`` and ``absolute_value_applied`` columns identify
which replacement was applied to each mode.

``uncertainty_eigenmode_mts`` attributes each modal contribution to the
selected MTs. Split the eigenvector and sensitivity according to the energy
groups belonging to MT :math:`m`, and define

.. math::

   a_{k,m} = \mathbf{q}_{k,m}^{\mathrm{T}}\mathbf{S}_m.

The table reports the diagonal part
:math:`d_{k,m}=\lambda_k a_{k,m}^2`. Each unordered cross term between two MTs
is divided equally between them. The cross part assigned to MT :math:`m` and
the resulting attribution are therefore

.. math::

   c_{k,m} = \lambda_k a_{k,m}\sum_{n\ne m}a_{k,n},
   \qquad
   t_{k,m} = d_{k,m}+c_{k,m}
           = \lambda_k a_{k,m}\sum_n a_{k,n}.

This convention gives the exact closure
:math:`\sum_m t_{k,m}=v_k`. It is an accounting convention rather than a
unique physical allocation: diagonal, cross-allocated, and total-attributed
columns are kept separate for that reason. An individual MT attribution or
cross allocation can be negative even when the reconstructed full covariance
is PSD; only the complete contribution of a retained eigenmode is guaranteed
non-negative.

Both tables contain ``raw_*`` and ``used_*`` columns. This makes it possible to
compare the original evaluation with the matrix actually propagated:

.. code-block:: python

   uncertainty = Sandwich(
       sensitivities,
       covmat=covariances,
       list_za=[922380],
       list_MFs=[33],
       list_MTs=[2, 18, 102],
       non_psd_policy="eigen_decomp",
   )

   modes = uncertainty.uncertainty_eigenmodes
   mt_attribution = uncertainty.uncertainty_eigenmode_mts
   u238_mf33 = modes.xs(("U238", 33), level=("ZA", "MF"))

``covariance_diagnostics`` provides one row per available selected ZA/MF block,
including the selected MT tuple, matrix dimension, minimum eigenvalue,
tolerance, number of significant negative eigenvalues, number of negative
eigenvalues set to zero or replaced by their absolute values, selected
negative-eigenvalue policy, asymmetry norm, policy, and repair flag.

Serpent covariance-channel matching
-----------------------------------

When the sensitivity object comes from Serpent, ``Sandwich`` matches
covariances to sensitivities by covariance ``(MF, MT)`` channel rather than by
MT alone. This avoids ambiguous cases where the same MT value appears in
different ENDF files. For example:

* MF=33, MT=18 is matched to the fission sensitivity ``fission``.
* MF=35, MT=18 is matched to ``chi prompt``.
* MF=31, MT=452/455/456 is matched to the corresponding nubar sensitivity.
* MF=34, MT=251 is matched to ``ela leg mom 1``.
* MF=34, MT=2 is treated as the currently supported ``L=1`` angular block.

The MF34 angular-distribution treatment is deliberately limited. pyNDUS
currently propagates only the first elastic Legendre moment
:math:`L=1` (:math:`\bar{\mu}=a_1`). This may come from the reduced
MF34/MT251 representation or from an MF34/MT2 covariance. In both cases the
matched sensitivity is ``ela leg mom 1``.

Higher Legendre moments are not propagated at present. If only
``ela leg mom 2`` or higher can be resolved for an MF34 covariance, pyNDUS
raises an explicit error instead of silently pairing the covariance with the
wrong sensitivity. Analyses that require ``ela leg mom 2`` or higher need an
L-resolved covariance representation.

Selections are interpreted against both the physical quantity and the stored
covariance MT. Therefore ``list_MTs=[2]`` can select an available
MF34/MT251 block for ``ela leg mom 1``; ``list_MTs=[251]`` can select a stored
MF34/MT2 block for the same first Legendre moment. Result tables keep the MT
actually read from the covariance matrix. Use ``list_MFs=[34]`` or
``list_MFs=[35]`` when these non-default covariance files should be included.

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
the two normalization terms. This includes the selected ``non_psd_policy``:
``warn`` retains the original covariance everywhere, ``raise`` rejects an
indefinite selected block before calculating the coefficient, and
``eigen_decomp`` uses the same reconstructed PSD block, including the selected
``neg_eigval_policy``, in all three terms.

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
