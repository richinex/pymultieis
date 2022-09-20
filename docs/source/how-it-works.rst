=========================================
How :code:`pymultieis` works
=========================================

The batch fitting algorithm implemented in :code:`pymultieis` is described in the paper
On the Analysis of Non-stationary Impedance Spectra by Alberto Battistel, Guoqing Du, and Fabio La Mantia.
Fitting is done via complex non-linear optimization of the model parameters using two approaches - deterministic and stochastic.
The deterministic optimization uses the BFGS/L-BFGS routines provided by the Pytorch-minimize package
which uses the real first and second derivatives computed via automatic differentiation.
The stochastic option uses Adam optimizer from the torch.optim.Optimizer class.

Rather than rely on the use prefit and use previous approach to batch-fitting,
the algorithm implemented in pymultieis preserves the correlation between parameters by introducing a custom cost function
which is a combination of the scaled version of the chisquare used in complex nonlinear regression and two additional terms:

- Numerical integral of the second derivative of the parameters with respect to the immittance and
- A smoothing factor.

Minimizing these additional terms allow the algorithm to minimize the number of times the curve changes concavity.
This allows the minimization algorithm to obtain smoothly varying optimal parameters.


