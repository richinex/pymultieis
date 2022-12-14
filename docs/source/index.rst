.. pymultieis documentation master file, created by
   sphinx-quickstart on Wed Sep 14 21:21:29 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

pymultieis
=============
.. code-block:: bash

   "Simplicity is the ultimate sophistication - Leonardo da Vinci"

:code:`pymultieis` offers a robust approach to batch-fitting electrochemical impedance spectra with a dependence.
Dependence implies that the spectra being fitted are gradually varying or similar to each other
and were obtained as a result of continuous change of in the property of the electrochemical system under study.
Such properties include but are not limited to temperature, potential, state of charge and depth of discharge.

The batch-fitting algorithm implemented in pymultieis allows the kinetic parameters of the system
such as the charge transfer resistance, double layer capacitance and Warburg coefficient to be obtained
as curves which vary as a function of the dependent variable under study.

The ``py`` in ``pymultieis`` represents python while the ``multieis`` is an abbreviation for ``Multiple Electrochemical Impedance Spectra``.

:code:`pymultieis` offers methods modules for model fiting, model validation, visualization,


Installation
------------

.. code-block:: bash

   pip install pymultieis

:ref:`quick-start-guide-label` contains a step-by-step tutorial
on getting started with :code:`pymultieis`.

Dependencies
~~~~~~~~~~~~

pymultieis requires:

-   Python (=3.9)
-   `torch <https://pytorch.org/get-started/locally/>`_ (>=1.13.0)
-   `pytorch-minimize <https://pytorch-minimize.readthedocs.io/en/latest/install.html>`_
-   Matplotlib (>=3.6.0)
-   NumPy (>=1.23.3)
-   Pandas (>=1.4.4)


Examples and Documentation
---------------------------

:ref:`quick-start-guide-label` contains a detailed guide on getting started with :code:`pymultieis`.
It is assumed that the user is already familiar with basic python syntax.
Detailed tutorials on several aspects of :code:`pymultieis` can be found in the :code:`examples/` directory.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   quick-start-guide
   pymultieis
   examples
   simultaneous-vs-sequential-fit
   how-it-works
   troubleshooting
   extra-resources
   FAQ




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
