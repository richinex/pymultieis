.. _FAQ-label:

===================================================
Frequently asked questions
===================================================

1. How do I obtain the standard deviation of my impedance measurements.

2. What if I have just one spectra and want to fit it

While ``pymultieis`` is meant to be used for a sequence of spectra, It can also be tweaked to fit a single spectra.
The trick is to repeat the single spectra up to a certain number, say 10 and use ``fit_simultaneous()`` or ``fit_stochastic()``
with the smoothing factor for all parameters set to ``inf``. For instance:

.. code-block:: python

  Y_single_spectra = Y_her[:, 40]
  Y_single_spectra.shape
  # torch.Size([35])

  Y_repeated = torch.tile(Y_her_single_spectra[:,None], (1, 10))
  Y_repeated.shape
  # torch.Size([35, 10])

  smf = torch.full((len(p0),), torch.inf)
  eis_her = pym.Multieis(p0, F_her, Y_repeated, bounds, smf, her, weight= 'modulus', immittance='admittance')
  popt, perr, chisqr, chitot, AIC = eis_her.fit_stochastic()
  popt, perr, chisqr, chitot, AIC = eis_her.fit_sequential()

See the notebook ``smoothing-factor-effect.ipynb`` under `Examples <https://github.com/richinex/pymultieis/tree/main/docs/source/examples>`_ for more details.


