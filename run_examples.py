import numpy as np
from np_array_on_custom_type import njit_fill_target, njit_without_jitclass, njit_with_jitclass

if __name__ == "__main__":
    result = njit_fill_target(np.empty(60))
    print("fill a target array works", result.an_array)

    no_jitclass_result = njit_without_jitclass()
    print('NOT using a jit class returns all zeros', no_jitclass_result.an_array)

    jitclass_result = njit_with_jitclass()
    print('using a jit class works', jitclass_result.an_array)
