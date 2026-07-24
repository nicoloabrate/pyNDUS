Quick start
===========

The following examples illustrate the intended high-level workflow. File names,
energy grids, nuclear-data libraries, and available responses must be adapted
to the user's calculations.

Read a sensitivity file
-----------------------

.. code-block:: python

   from pyNDUS import Sensitivity

   sens = Sensitivity("case_sens0.m")
   print(sens.responses)
   print(sens.MTs)

Multiple Serpent sensitivity files can be merged into a single object when
their energy-group structures are consistent:

.. code-block:: python

   sens = Sensitivity(
       ["case_a_sens0.m", "case_b_sens0.m"],
       duplicate_policy="raise",
   )

The available duplicate policies are ``"raise"``, ``"keep_first"``, and
``"keep_last"``.

Extract a sensitivity profile
-----------------------------

.. code-block:: python

   profile = sens.get(
       resp=["keff"],
       mat=["total"],
       za=[922350],
       MT=[18],
       group_order="ascending",
   )

When relative standard deviations are available, ``get`` returns both the
mean sensitivity profile and its relative standard deviation.

Read a covariance matrix
-----------------------

.. code-block:: python

   from pyNDUS import Covariance

   covar_path = "/home/user/covariance_NEWCOV"
   T = 300 # K
   energy_grid_structure = [1.00001e-11, 1.00000e-07, 5.40000e-07, 4.00000e-06, 8.31529e-06, 1.37096e-05,
                            2.26033e-05, 4.01690e-05, 6.79040e-05, 9.16609e-05, 1.48625e-04, 3.04325e-04,
                            4.53999e-04, 7.48518e-04, 1.23410e-03, 2.03468e-03, 3.35463e-03, 5.53084e-03,
                            9.11882e-03, 1.50344e-02, 2.47875e-02, 4.08677e-02, 6.73795e-02, 1.11090e-01,
                            1.83156e-01, 3.01974e-01, 4.97871e-01, 8.20850e-01, 1.35335e+00, 2.23130e+00,
                            3.67879e+00, 6.06531e+00, 1.00000e+01, 1.96403e+01,]
   cov_Pu239 = Covariance("Pu-239", T, energy_grid_sens, database=True,
                     egridname="ECCO-33", lib="endfb_80", cwd=covar_path,)


Extract a covariance sub-matrix
-----------------------------

.. code-block:: python

    cov_fission = cov_Pu239.get(18, MF=33)
    cov_capture_fission = cov_Pu239.get((102, 18), MF=33)
    cov_pair = cov_Pu239.get([18, 102], MF=33)

MF section is a mandatory argument to select the proper MT-MF couple.

Run a sandwich calculation
--------------------------

.. code-block:: python

   from pyNDUS import Sandwich

   result = Sandwich(
       sens=sens,
       covmat=covariances,
       list_resp=["keff"],
       list_mat=["total"],
   )

The default Sandwich ``calculation_type`` is ``uncertainty`` but also ``representativity`` and ``similariy`` are available.
The covariance dictionary must contain compatible ``Covariance`` objects.
See :doc:`user_guide/sandwich` for the calculation modes and missing-covariance
policies. 