import numpy as np
from np_array_on_custom_type import njit_fill_target, njit_with_jitclass

if __name__ == "__main__":
    result = njit_fill_target(np.empty(60))
    assert (result.an_array == np.ones(60)).all()
    print("fill a target array works", result.an_array)

    jitclass_result = njit_with_jitclass()
    print('using a jit class works', jitclass_result.an_array)