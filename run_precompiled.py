import numpy as np
from precompile import precompile_fails, precompile_works_with_target

if __name__ == "__main__":
    result = precompile_works_with_target(np.empty(60))
    print("works with a target array", result.an_array)

    fails = precompile_fails()