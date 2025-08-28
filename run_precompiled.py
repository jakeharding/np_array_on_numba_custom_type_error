import numpy as np
from precompile import precompile_no_target, precompile_works_with_target

if __name__ == "__main__":
    result = precompile_works_with_target(np.empty(60))
    print("works with a target array", result.an_array)

    no_target_result = precompile_no_target()
    print("no target result", no_target_result.an_array)
    assert (no_target_result.an_array == np.ones(60) * 5).all()