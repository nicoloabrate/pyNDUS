Sensitivity Channels API
========================

``SensitivityChannel`` is the canonical identifier for one sensitivity-profile
axis entry. It keeps average-side ENDF identifiers, covariance-side ENDF
identifiers, and the optional Legendre order ``L`` together.

Use ``from_alias`` when starting from reader labels:

.. code-block:: python

   channel = SensitivityChannel.from_alias("chi prompt")
   scattering = SensitivityChannel.from_alias("ela leg mom 1")

Use ``from_endf`` when starting from ENDF identifiers. Either the average side
or the covariance side may be provided when the result is unique:

.. code-block:: python

   fission_xs = SensitivityChannel.from_endf(average_MF=3, average_MT=18)
   chi_prompt = SensitivityChannel.from_endf(covariance_MF=35, covariance_MT=18)
   elastic_p1 = SensitivityChannel.from_endf(covariance_MF=34,
                                             covariance_MT=2,
                                             L=1)

MF/MT pairs are not always sufficient. For example, MF34/MT2 may refer to
several Legendre moments, so ``L`` is required. pyNDUS raises a ``ValueError``
instead of guessing.

.. automodule:: pyNDUS.channels
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:
