.. _how-it-works-label:

=========================================
How :code:`pymultieis` works
=========================================

The fitting algorithm implemented in :code:`pymultieis` is described in the `paper <https://doi.org/10.1002/elan.201600260>`_
by Alberto Battistel, Guoqing Du, and Fabio La Mantia.
Fitting is done via complex non-linear optimization of the model parameters using two approaches - deterministic and stochastic.
The deterministic optimization is based on the TNC/BFGS/L-BFGS solvers from the `Optimizer API <https://pytorch-minimize.readthedocs.io/en/latest/api/index.html#optimizer-api>`_
provided by the `Pytorch-minimize <https://github.com/rfeinman/pytorch-minimize>`_ library.
The stochastic option uses `Adam <https://doi.org/10.48550/arXiv.1412.6980>`_ from the `torch.optim.Optimizer <https://pytorch.org/docs/stable/optim.html>`_ class.

Rather than rely on the use prefit and use previous approach to batch-fitting,
the algorithm implemented in pymultieis preserves the correlation between parameters by introducing a custom cost function
which is a combination of the scaled version of the chisquare used in complex nonlinear regression and two additional terms:

- Numerical integral of the second derivative of the parameters with respect to the immittance and
- A smoothing factor.

Minimizing these additional terms allow the algorithm to minimize the number of times the curve changes concavity.
This allows the minimization algorithm to obtain parameters which smoothly vary as a function of the sequence index.
