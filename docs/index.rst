pyNDUS documentation
====================

``pyNDUS`` is an open-source Python package for nuclear-data sensitivity,
uncertainty, similarity, and representativity calculations. It combines
multi-group sensitivity profiles produced by Serpent or ERANOS with relative
covariance matrices generated from ENDF-6 evaluations through NJOY and SANDY.

The package provides three main capabilities:

* reading and manipulating sensitivity profiles;
* processing and extracting multi-group covariance matrices;
* evaluating uncertainty, similarity, and representativity quantities.

.. note::

   ``pyNDUS`` is research software under active development. Numerical results
   should be supported by appropriate verification, validation, and input-data
   quality checks for the intended application.

Documentation structure
-----------------------

The :doc:`user_guide/overview` introduces the package workflow. The
:doc:`theory` page summarizes the mathematical definitions. The
:doc:`development/testing` page documents the current verification strategy
and the logic of the automated tests. The :doc:`api/index` section is generated
from the source-code docstrings.

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: User guide

   user_guide/overview
   user_guide/sensitivity
   user_guide/covariance
   user_guide/sandwich
   theory

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: Development

   development/testing
   development/contributing

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api/index
