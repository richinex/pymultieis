.. _extra-resources-label:

===================================================
Extra Resources
===================================================


Distributed Elements
===================================================

Semi-infinite (planar infinite length) Warburg
***************************************************

Describes linear diffusion from a medium with lebgth which can be approximated
as infinite.

.. math:: Z_{W} = \frac{A_W}{\sqrt{w}}(1-j)

Or

.. math::
    Z_{W} = \sqrt{\frac{R_d}{s C_d}} (\sqrt{R_{d}~sC_{d}})

Where :math:`s = j \omega` with :math:`j` being the imaginary unit and :math:`\omega` the angular frequency.
:math:`A_{W}` has units of :math:`\Omega s^{-0.5}`, :math:`R` has units of Ohms (:math:`\Omega`) and :math:`C` has units of Farads (:math:`F`).
And

.. math::
    A_W = \frac{RT}{F^{2}C_{o}\sqrt{D_o}}

.. code-block:: python

  w = 2 * torch.pi * freq
  s = 1j * w
  Zw = Aw/torch.sqrt(w) * (1-1j)

  # Or

  Zw = torch.sqrt(Rd/s*Cd) * (torch.sqrt(Rd * s*Cd))


Finite length diffusion with reflective boundary
*****************************************************

Describes the reaction of mobile active species distributed in a layer with finite length,
terminated by an impermeable boundary.

.. math:: Z_{Wo} = \sqrt{\frac{R_d}{s C_d}} \coth(\sqrt{R_{d}~sC_{d}})

Or

.. math:: Z_{Wo} = R \frac{coth(j \omega \tau)^{\phi}}{(j \omega \tau)_{\phi}}



Where :math:`\phi` = 0.5

.. code-block:: python

  w = 2 * torch.pi * freq
  s = 1j * w
  ZWs = torch.sqrt(Rd/s*Cd) * 1/torch.tanh(torch.sqrt(Rd * s*Cd))


Finite length diffusion with transmissive boundary
******************************************************

Describes the reaction of mobile active species distributed in a layer with finite length,
terminated by an impermeable boundary.

.. math:: Z_{Ws} = \sqrt{\frac{R_d}{s C_d}} \tanh(\sqrt{R_{d}~sC_{d}})

Or

.. math:: Z_{Ws} = R \frac{tanh(j \omega \tau)^{\phi}}{(j \omega \tau)_{\phi}}


Where :math:`\phi` = 0.5

.. code-block:: python

  w = 2 * torch.pi * freq
  s = 1j * w
  ZWs = torch.sqrt(Rd/s*Cd) * torch.tanh(torch.sqrt(Rd * s*Cd))


Resources on the web
===================================================
1. `Research Solutions and Resources LLC <http://www.consultrsr.net/resources/eis/>`_ has a section on electrochemical impedance spectroscopy (EIS)
which explains several concepts related to the study of impedance such as fitting equivalent circuits to EIS data, the constant phase element (CPE),
diffusion, porous electrodes to mention a few. There are also links to several other resources.

2. `Matt Lacey's website <http://lacey.se/science/eis/diffusion-impedance/>`_ provides an excellent description of diffusion impedance.

Books
===========

1. `Electrochemical Impedance Spectroscopy <https://www.wiley.com/en-us/Electrochemical+Impedance+Spectroscopy,+2nd+Edition-p-9781118527399>`_ by Mark Orazem and Bernard Tribollet

2. `Electrochemical Impedance Spectroscopy and its Applications <https://link.springer.com/book/10.1007/978-1-4614-8933-7>`_ by Andrzej Lasia

Videos
========

1. Sam Cooper's `Introduction to Electrochemical Impedance Spectroscopy (EIS: Maths and Theory) <https://www.youtube.com/watch?v=5puDQjCl2pk>`_

2. S. Ramanathan's `NPTEL-NOC IITM lectures <https://www.youtube.com/watch?v=_bRI2bv_YqY&list=PLyqSpQzTE6M9ftJKyUWBilfrgBjh_6eh1>`_

3. Across the Nanoverse's `Introduction to EIS <https://www.youtube.com/watch?v=xaimI9w-egQ>`_