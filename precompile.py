import numpy as np
from numba import types
from numba.pycc import CC

from np_array_on_custom_type import CustomResult, custom_result_type_instance

cc = CC("precompile")

@cc.export("precompile_fails", custom_result_type_instance())
def precompile_fails():
    """
    Occasionally will see a result returned but with all zeros or junk values.
    Fails with various errors depending on architecture.
    Seg Faults, Abort traps, double free errors
    """
    return CustomResult(np.ones(60) * 5)

@cc.export("precompile_works_with_target", custom_result_type_instance(types.float64[:]))
def precompile_works_with_target(target_array):
    """
    Works when filling an existing array. Maybe since a reference is held outside of the function?
    Precompilation won't succeed when using a jitclass.
    """
    target_array.fill(np.ones(60) * 5)
    return CustomResult(target_array)

if __name__ == "__main__":
    cc.compile()
    print("precompiled")