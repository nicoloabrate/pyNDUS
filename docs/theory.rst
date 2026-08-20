Mathematical background
=======================

Sensitivity coefficients
------------------------

A relative sensitivity coefficient of a response :math:`R` to a parameter
:math:`p_i` is commonly written as

.. math::

   S_i = \frac{p_i}{R}\frac{\partial R}{\partial p_i}.

In a multi-group treatment, the sensitivity profile is represented by a vector
whose components correspond to energy groups. Multiple nuclear-data parameters
are identified by nuclide and ENDF reaction identifiers.

Uncertainty propagation
-----------------------

For a sensitivity vector :math:`\mathbf{S}` and a relative covariance matrix
:math:`\mathbf{C}`, first-order uncertainty propagation follows the sandwich
rule

.. math::

   \sigma_R^2 = \mathbf{S}^{\mathrm{T}}\mathbf{C}\mathbf{S}.

For several reactions, the complete result includes diagonal contributions and
cross-reaction terms

.. math::

   q_{ij} =
   \mathbf{S}_i^{\mathrm{T}}\mathbf{C}_{ij}\mathbf{S}_j.

These terms are variance contributions. For display, pyNDUS can apply the
sign-preserving square-root convention

.. math::

   \widetilde{q}_{ij} =
   \operatorname{sign}(q_{ij})\sqrt{|q_{ij}|}.

In particular, a negative covariance contribution is displayed as
:math:`-\sqrt{|q_{ij}|}`. These signed roots must not be summed to obtain the
total uncertainty. The response standard deviation is instead

.. math::

   \sigma_R = \sqrt{\sum_{ij}q_{ij}},

so the original signed variance and covariance contributions are summed before
the square root is taken.

Similarity
----------

The similarity between sensitivity profile :math:`i` from the first system and
profile :math:`j` from the second system is evaluated using the normalized
scalar product

.. math::

   E_{ij} =
   \frac{\mathbf{S}_{1,i}^{\mathrm{T}}\mathbf{S}_{2,j}}
   {\sqrt{\mathbf{S}_{1,i}^{\mathrm{T}}\mathbf{S}_{1,i}}
    \sqrt{\mathbf{S}_{2,j}^{\mathrm{T}}\mathbf{S}_{2,j}}}.

It measures geometric alignment of the profiles but does not weight individual
nuclear-data parameters by their covariance. Each coefficient is normalized
with the same two vectors appearing in its numerator; norms are never
accumulated globally over other MT reactions. Therefore, for non-zero vector
norms, :math:`-1 \leq E_{ij} \leq 1`, and an identical profile compared with
itself has :math:`E_{ii}=1`.

Representativity
----------------

The covariance-weighted representativity coefficient is

.. math::

   r = \frac{\mathbf{S}_1^{\mathrm{T}}\mathbf{C}\mathbf{S}_2}
   {\sqrt{\mathbf{S}_1^{\mathrm{T}}\mathbf{C}\mathbf{S}_1}
    \sqrt{\mathbf{S}_2^{\mathrm{T}}\mathbf{C}\mathbf{S}_2}}.

For identical systems and a non-zero denominator, :math:`r=1`. Unlike the
unweighted similarity coefficient, representativity incorporates the relative
importance and correlation structure of the nuclear-data uncertainties.

In pyNDUS the displayed ZA/MF/MT matrix contains additive contributions to this
single globally normalized coefficient. The total representativity is

.. math::

   r_{\mathrm{total}} = \sum_{z,i,j} r_{z,ij}.

When the isotope sets differ, absent sensitivity profiles are treated as zero
in the numerator, so only their intersection contributes there. The
normalization terms remain the complete uncertainty of each system: isotopes
exclusive to the first or second system contribute respectively to
:math:`\mathbf{S}_1^{\mathrm{T}}\mathbf{C}\mathbf{S}_1` or
:math:`\mathbf{S}_2^{\mathrm{T}}\mathbf{C}\mathbf{S}_2`.

ERRORR MF and MT organization
-----------------------------

``pyNDUS`` groups reaction identifiers (MT) by ERRORR section (MF). For
example, cross-section covariances are generally associated with MF=33, while
neutron-multiplicity covariances are associated with MF=31. A given normalized
MT is assigned to one MF section in the package mapping, avoiding ambiguous
accumulation of the same parameter across multiple MF sections.

For common ENDF quantities, the covariance file is not the same as the file
that stores the mean quantity:

.. list-table::
   :header-rows: 1

   * - Quantity file
     - Quantity
     - Covariance file
   * - MF=1
     - neutron multiplicities, such as MT=452, 455, 456
     - MF=31
   * - MF=3
     - cross sections
     - MF=33
   * - MF=4
     - angular distributions
     - MF=34
   * - MF=5
     - energy distributions, such as fission spectra
     - MF=35

This distinction is especially important for Serpent perturbations that are
identified by names rather than by a unique ENDF MT. ``chi prompt`` is matched
to MF=35, MT=18, while ``xs 18`` is matched to MF=33, MT=18.

MF34 and the reduced MT251 angular covariance
---------------------------------------------

MT=251 is the average scattering cosine, :math:`\bar{\mu}`. For an angular
distribution represented by Legendre coefficients,

.. math::

   f(\mu) = \sum_l \frac{2l+1}{2} a_l P_l(\mu),

the first coefficient satisfies :math:`\bar{\mu}=a_1`. Therefore an
MF34/MT251 covariance is a reduced covariance for the first elastic Legendre
moment only. pyNDUS maps this block to the Serpent perturbation
``ela leg mom 1``.

The current implementation does not propagate Legendre moments beyond
:math:`L=1`. Full MF34/MT2 covariance data can in principle contain several
Legendre orders, but pyNDUS covariance matrices are currently indexed by MF
and MT only, not by :math:`L`. If such a covariance channel is requested and
would require choosing between multiple Legendre sensitivities, pyNDUS raises
an explicit error rather than assuming the wrong moment. Support for
``ela leg mom 2`` and higher requires an L-resolved MF34 covariance
representation.
