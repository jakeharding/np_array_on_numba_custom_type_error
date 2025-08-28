import sys
import numpy as np
from np_array_on_custom_type import works_example, error_example

if __name__ == "__main__":
    result = works_example(np.empty(60))
    assert (result.an_array == np.ones(60)).all()
    print("from working", result.an_array)

    if len(sys.argv) > 1:
        err = error_example()
        print('from error', err.an_array)