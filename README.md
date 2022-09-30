<div align="center">
<img src="https://github.com/richinex/pymultieis/blob/main/docs/source/_static/z_bode.png" alt="logo"></img>
</div>


pymultieis
=============

[**Installation**](#installation)
| [**Examples**](https://github.com/richinex/pymultieis/tree/main/docs/source/examples)
| [**Documentation**](https://pymultieis.readthedocs.io/en/latest/index.html)
| [**Citing this work**](#citation)


A library for fitting a sequence of electrochemical impedance spectra.

- Implements algorithms for simultaneous and sequential fitting.

- Written in python and based on the [PyTorch](https://pytorch.org/) library.

- Leverages deterministic solvers from [pytorch-minimize](https://pytorch-minimize.readthedocs.io/en/latest/api/index.html) which compute the first- and second-order derivatives via autograd.

## Installation<a id="installation"></a>

pymultieis requires the following:

-   Python (>=3.9)
-   [torch](https://pytorch.org/get-started/locally/) (>=1.12.1)
-   [pytorch-minimize](https://pytorch-minimize.readthedocs.io/en/latest/install.html)
-   Matplotlib (>=3.6.0)
-   NumPy (>=1.23.3)
-   Pandas (>=1.4.4)
-   SciPy (>=1.9.1)


After installing the dependencies, you can now install pymultieis via the following pip command

```
pip install pymultieis
```

[Getting started with pymultieis](https://pymultieis.readthedocs.io/en/latest/quick-start-guide.html#) contains a quick start guide to
fitting your data with ``pymultieis``.


## Examples

Detailed tutorials on other aspects of ``pymultieis`` can be found in [Examples](https://github.com/richinex/pymultieis/tree/main/docs/source/examples).

## Documentation

Details about the ``pymultieis`` API, can be found in the [reference documentation](https://pymultieis.readthedocs.io/en/latest/index.html).


## Citing this work<a id="citation"></a>

If you use pymultieis for academic research, you may cite the library as follows:

```
@misc{Chukwu2021,
  author = {Chukwu, Richard},
  title = {pymultieis: a library for fitting a sequence of electrochemical impedance spectra},
  publisher = {GitHub},
  year = {2022},
  url = {https://github.com/richinex/pymultieis},
}
```