Reproduceable example of an error in numba njit functions that return custom type objects with numpy arrays when the arrays are created in the njit function. 

Error is not seen when the arrays are created outside of the njit function and filled inside the function.

Tested using Python 3.13

```bash
pip install numba setuptools
python -m run_examples error
python -m precompile
python -m run_precompiled
```
