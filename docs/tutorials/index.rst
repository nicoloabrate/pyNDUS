Tutorial notebooks
==================

These notebooks are executable tutorials built around the local benchmark data
used during pyNDUS development. They are also useful as smoke tests: the code
cells contain assertions on the expected reader behavior and data shapes.

The notebooks look for data in ``PYNDUS_TUTORIAL_DATA`` first, then fall back to
the local development path used to create them. Documentation builds render the
notebooks without executing them.

.. toctree::
   :maxdepth: 1

   sensitivity_serpent
   sensitivity_eranos
   sensitivity_algebra
   covariance
   sandwich
