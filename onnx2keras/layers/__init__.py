from typing import Callable

from .upsample import convert_upsample
from .resize import convert_resizing

AVAILABLE_CONVERTERS = {}


def register_converter(converter: Callable, op_name: str, op_type: str):
    if op_name in AVAILABLE_CONVERTERS:
        raise ValueError("Converter for op type {} already registered.".format(op_name))
    AVAILABLE_CONVERTERS[op_name] = converter


AVAILABLE_CONVERTERS = {
    "Upsample": convert_upsample,
    "Resize": convert_resizing,
}
