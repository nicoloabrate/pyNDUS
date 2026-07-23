Contributing
============

Development setup
-----------------

Create an isolated environment, clone the repository, and install the package
in editable mode:

.. code-block:: console

   git clone https://github.com/nicoloabrate/pyNDUS.git
   cd pyNDUS
   python -m pip install -e .
   python -m pip install -r docs/requirements.txt

Before submitting changes, run:

.. code-block:: console

   python -m pytest
   sphinx-build -W -b html docs docs/_build/html

Numerical changes
-----------------

Changes to uncertainty, similarity, representativity, MF/MT mapping, or profile
merging should include at least one focused unit test. If a validated numerical
output changes intentionally, update the relevant regression value and explain
the mathematical or data-model reason in the change description.

Documentation changes
---------------------

Public classes and methods use NumPy-style docstrings. User-visible features
should be documented in the user guide in addition to the API docstring.
