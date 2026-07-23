Workflow overview
=================

A typical ``pyNDUS`` calculation consists of four stages:

#. Generate sensitivity profiles with Serpent or ERANOS.
#. Obtain evaluated nuclear-data files in ENDF-6 format.
#. Process and parse multi-group covariance matrices with NJOY and SANDY.
#. Combine sensitivities and covariances using uncertainty, similarity, or
   representativity calculations.

The three public classes correspond to these stages:

``Sensitivity``
   Reads, stores, extracts, collapses, and optionally merges sensitivity
   profiles.

``Covariance``
   Processes or reads ERRORR covariance data and exposes covariance blocks by
   MF and MT identifiers.

``Sandwich``
   Combines sensitivity profiles and covariance matrices and stores the
   requested numerical result.

The energy-group structure and ordering must be consistent between sensitivity
profiles and covariance matrices. Multi-file sensitivity merging explicitly
checks the group boundaries before combining the input data.
