# region standard imports

from typing import (
    Generic,
    TypeVar,
    List,
    Dict,
    Optional,
    Callable,
    Any,
    cast,
    Iterator,
    Generator,
    Iterable,
    Type,
    TYPE_CHECKING,
)
from abc import ABC, ABCMeta, abstractmethod
from array import array
import numpy
import ctypes


# endregion


# region custom imports
from utils.exceptions import *
from utils.exceptions import *


if TYPE_CHECKING:
    from utils.custom_types import T
    from adts.linked_list_adt import LinkedListADT, iNode
    from adts.stack_adt import StackADT
    from adts.collection_adt import CollectionADT


# endregion

class StackUtils:
    def __init__(self, stack_obj: "StackADT[T]") -> None:
        self.obj = stack_obj

    def check_underflow_error(self):
        """if the stack is empty raises an error."""
        if self.obj.top is None:
            raise DsUnderflowError(f"Error: The Stack is empty.")

    def check_array_stack_underflow_error(self):
        if self.obj.top == -1:
            raise DsUnderflowError(f"Error: The Stack is empty.")
        if self.obj.is_empty():
            raise DsUnderflowError(f"Error: The Stack is empty.")

    def min_max_standard_comparison_lib(self):
        """Default Comparisons for major datatypes - required for the Min Max Stack. Also takes a custom key - a function for comparing the stack items."""
        if issubclass(self.obj.datatype, (int, float, complex, tuple)):
            # Python tuples are compared lexicographically → multi-level comparison is automatic.
            # Strings are iterable, but often you want lexicographical comparison (default) instead of length.
            return lambda x: x  # value is compared by numerical size.
        elif issubclass(self.obj.datatype, (list, dict, set, str)):
            return lambda x: len(x) # compare by number of elements
        else:
            raise KeyInvalidError(f"Error: Must provide a key in order to compare min and max effectively!")
