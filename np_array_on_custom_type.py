from contextlib import ExitStack

import numba
import numpy as np
import numpy.typing as npt
from numba import types
from numba.extending import (
    type_callable,
    as_numba_type,
    register_model,
    models,
    make_attribute_wrapper,
    typeof_impl,
    lower_builtin,
    unbox,
    NativeValue,
    box,
)
from numba.core import cgutils


class CustomResult:
    an_array: npt.NDArray[np.float64]
    def __init__(self, an_array: npt.NDArray[np.float64]):
        self.an_array = an_array


class CustomResultType(types.Type):
    def __init__(self):
        super().__init__(name="CustomObjectType")


custom_result_type_instance = CustomResultType()

@typeof_impl.register(CustomResult)
def typeof_custom_object_type(*_) -> types.Type:
    return custom_result_type_instance


as_numba_type.register(CustomResult, custom_result_type_instance)

@type_callable(CustomResult)
def type_custom_result(*_):
    def typer(an_array):
        if isinstance(an_array, types.Array):
            return custom_result_type_instance
        return None
    return typer

make_attribute_wrapper(CustomResultType, "an_array", "an_array")

@lower_builtin(CustomResult, types.Array)
def custom_result_impl(context, builder, sig, args):
    an_array = args[0]
    result = cgutils.create_struct_proxy(sig.return_type)(context, builder)
    result.an_array = an_array
    return result._getvalue()


@register_model(CustomResultType)
class CustomResultTypeModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ("an_array", types.float64[:])
        ]
        super().__init__(dmm, fe_type, members)


@unbox(CustomResultType)
def unbox_custom_result(typ, obj, c):
    is_error_ptr = cgutils.alloca_once_value(c.builder, cgutils.false_bit)
    result = cgutils.create_struct_proxy(typ)(c.context, c.builder)
    with ExitStack() as stack:
        an_array_obj = c.pyapi.object_getattr_string(obj, "an_array")
        with cgutils.early_exit_if_null(c.builder, stack, an_array_obj):
            c.builder.store(cgutils.true_bit, is_error_ptr)
        an_array_native = c.unbox(types.float64[:], an_array_obj)
        c.pyapi.decref(an_array_obj)
        with cgutils.early_exit_if(c.builder, stack, an_array_native.is_error):
            c.builder.store(cgutils.true_bit, is_error_ptr)
        result.cdf = an_array_native.value
    return NativeValue(result._getvalue(), is_error=c.builder.load(is_error_ptr))

@box(CustomResultType)
def box_custom_result(typ, val, c):
    ret_ptr = cgutils.alloca_once(c.builder, c.pyapi.pyobj)
    fail_obj = c.pyapi.get_null_object()
    with ExitStack() as stack:
        result = cgutils.create_struct_proxy(typ)(c.context, c.builder, value=val)
        an_array_obj = c.box(types.float64[:], result.an_array)
        with cgutils.early_exit_if_null(c.builder, stack, an_array_obj):
            c.pyapi.decref(an_array_obj)
            c.builder.store(fail_obj, ret_ptr)
        class_obj = c.pyapi.unserialize(c.pyapi.serialize_object(CustomResult))
        with cgutils.early_exit_if_null(c.builder, stack, class_obj):
            c.pyapi.decref(an_array_obj)
            c.builder.store(fail_obj, ret_ptr)
        res = c.pyapi.call_function_objargs(class_obj, (an_array_obj,))
        c.pyapi.decref(an_array_obj)
        c.builder.store(res, ret_ptr)
    return c.builder.load(ret_ptr)


@numba.njit
def error_example() -> CustomResult:
    return CustomResult(np.ones(60))

@numba.njit
def works_example(target_array):
    target_array.fill(np.ones(60))
    return CustomResult(target_array)