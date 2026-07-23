Testing and numerical verification
==================================

Purpose
-------

The automated tests are designed to protect the numerical core, data-indexing
logic, and recently added robustness features. They use small deterministic
arrays and lightweight fake objects whenever possible, so that routine tests do
not require Serpent, NJOY, or large evaluated-data files.

Test categories
---------------

Unit tests
~~~~~~~~~~

Unit tests verify an individual function or narrowly scoped behavior in
isolation. They use analytically tractable inputs and directly compare the
output against a known formula or invariant.

Regression tests
~~~~~~~~~~~~~~~~

Regression tests freeze the expected numerical behavior of representative
small cases. They detect unintended changes in MF/MT assembly, reaction cross
terms, isotope separation, and normalization after code modifications.

Integration-oriented tests
~~~~~~~~~~~~~~~~~~~~~~~~~~

The multi-file sensitivity tests exercise the public ``Sensitivity``
constructor while monkeypatching ``serpentTools.read``. This verifies the
complete pyNDUS merge path without relying on external Serpent files.

Continuous integration
~~~~~~~~~~~~~~~~~~~~~~

The intended CI workflow runs the test suite for every push and pull request.
Future extensions may add code coverage, linting, and type checking after the
numerical suite is stable.

Current coverage by test module
-------------------------------

``test_utils.py``
~~~~~~~~~~~~~~~~~

The utility tests verify:

* ZAIS-to-ZAID conversion;
* ZAID-to-ZAIS conversion;
* preservation of nominal values in NumPy-to-uncertainties conversion;
* conversion of relative standard deviations to absolute uncertainty.

``test_sandwich_uncertainty.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The uncertainty tests cover:

* the single-MT quadratic form :math:`\mathbf{S}^T\mathbf{C}\mathbf{S}`;
* diagonal covariance matrices;
* two-MT cross terms in both directions;
* zero contribution when the corresponding sensitivity profile is absent;
* the ``zero``, ``raise``, and ``assume`` missing-covariance policies;
* diagonal fallback covariance;
* off-diagonal fallback with zero and non-zero assumed correlation;
* rejection of invalid policy names.

``test_sandwich_similarity_representativity.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These tests verify:

* self-similarity;
* agreement with the analytical normalized dot product;
* symmetry under exchange of the two sensitivity profiles;
* self-representativity for a known covariance matrix;
* undefined representativity for a fully missing covariance under the ``zero``
  policy;
* explicit failure under the ``raise`` policy;
* self-representativity equal to one under a consistently applied fallback
  covariance;
* agreement with the fallback-covariance analytical formula;
* rejection of invalid policy names.

``test_regression.py``
~~~~~~~~~~~~~~~~~~~~~~

The regression cases freeze numerical behavior for increasingly complex
ERRORR layouts:

* a small two-group reference case;
* multiple MF sections with one MT and one ZA;
* multiple MF sections and MTs for one ZA;
* multiple MF sections and ZAs with one MT per section;
* multiple MF sections, multiple MTs, and multiple ZAs.

The MT sets are consistent with pyNDUS semantics: an MT is assigned to one MF
section. Cross terms are tested between MTs belonging to the same MF section,
and isotope blocks are verified independently.

``test_getsensitivity_multifile.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The multi-file tests cover:

* merging disjoint responses, nuclides, and MT profiles;
* rejection of inconsistent energy-group structures;
* duplicate detection with ``duplicate_policy="raise"``;
* retention of the first duplicate with ``keep_first``;
* retention of the last duplicate with ``keep_last``;
* passing the same file twice and obtaining an object identical to the original
  under ``keep_first`` or ``keep_last``;
* detection of duplicates when the same file is supplied twice under
  ``raise``;
* rejection of invalid duplicate-policy names;
* preservation of both mean profiles and relative standard deviations.

Running the tests
-----------------

Install the package in editable mode and execute:

.. code-block:: console

   python -m pip install -e .
   python -m pytest

Run an individual module with:

.. code-block:: console

   python -m pytest tests/test_sandwich_uncertainty.py -v

Recommended next developments
-----------------------------

The next testing steps are:

#. add CI through GitHub Actions;
#. publish a coverage report and prioritize uncovered numerical branches;
#. add minimal real-file fixtures for Serpent and ERRORR parsing;
#. add tests for energy-grid ordering between sensitivity and covariance data;
#. add executable tutorial examples when stable notebooks become available;
#. define benchmark cases against independently computed reference results.
