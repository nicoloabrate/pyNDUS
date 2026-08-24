Covariance matrices
===================

``Covariance`` manages multi-group relative covariance matrices obtained
from ENDF-6 nuclear-data evaluations. The class can coordinate processing with
SANDY and NJOY or read existing ERRORR output from a working directory.
With ``database=True``, ``cwd`` is treated as the root of a structured
library/grid tree. With ``database=False``, covariance and ENDF files are read
or generated directly in ``cwd``.

Existing ERRORR files are looked up using several filename conventions. The
historical pyNDUS/NJOY-generated name
``<ZAIS>_<temperature>K.errorr<MF>`` is still preferred, for example
``U-235_300K.errorr33``. If it is not present, pyNDUS also accepts
temperature-free names:

* ``<ZAIS>.errorr<MF>``, for example ``U-235.errorr33``;
* ``<ZAID>0.errorr<MF>``, for example ``9223500.errorr33``;
* ``<ZAID>.errorr<MF>``, for example ``922350.errorr33``.

This applies both to structured databases and to flat ``cwd`` directories.
When pyNDUS has to generate new ERRORR files, it still writes the historical
``<ZAIS>_<temperature>K.errorr<MF>`` name.

Core inputs include:

* ZAID;
* temperature;
* energy-group structure and its name;
* evaluated nuclear-data library;
* working or database directory;
* resonance-processing options.

The parsed covariance matrices are organized by ERRORR MF section and MT pair.
The ``get`` method extracts a requested covariance block and can return either
the underlying pandas representation or a NumPy array. The ``MF`` argument is
required, keyword-only, and must identify the section explicitly, either as an
integer such as ``MF=33`` or as an ERRORR key such as ``MF="errorr35"``.

.. code-block:: python

   from pyNDUS import Covariance

   cov = Covariance(
       zaid=922350,
       temperature=300,
       group_structure=energy_grid,
       egridname="custom",
       lib="endfb_80",
       cwd="covariance_database",
   )

   c_18_102 = cov.get((18, 102), MF="errorr33", to_numpy=True)

NJOY availability, evaluated-data access, and the exact SANDY interface are
external requirements and should be verified in the target environment.
