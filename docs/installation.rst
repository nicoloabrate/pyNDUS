Installation
============

Requirements
------------

``pyNDUS`` requires Python 3.8 or newer. Its main dependencies include NumPy,
SciPy, pandas, uncertainties, serpentTools, and SANDY. Covariance processing
may additionally require a working NJOY installation, depending on the chosen
workflow.

Installation from GitHub
------------------------

Clone the public repository and install the package from its root directory:

.. code-block:: console

   git clone https://github.com/nicoloabrate/pyNDUS.git
   cd pyNDUS
   python -m pip install .

For development, use an editable installation so that local source changes are
immediately visible:

.. code-block:: console

   python -m pip install -e .

Documentation dependencies
--------------------------

Install the documentation requirements with:

.. code-block:: console

   python -m pip install -r docs/requirements.txt

Then build the HTML documentation locally:

.. code-block:: console

   sphinx-build -W -b html docs docs/_build/html
