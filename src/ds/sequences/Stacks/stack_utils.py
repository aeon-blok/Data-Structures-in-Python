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


if TYPE_CHECKING:
    from user_defined_types.generic_types import T
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

    # region Min Max Avg
    def min_max_standard_comparison_lib(self):
        """Default Comparisons for major datatypes - required for the Min Max Stack. Also takes a custom key - a function for comparing the stack items."""
        if issubclass(self.obj.datatype, (int, float, str, tuple)):
            return lambda x: x  # value is compared by numerical size.
        elif issubclass(self.obj.datatype, (list, dict, set)):
            # Strings & tuples are iterable, but often you want lexicographical comparison (default) instead of length.
            return lambda x: len(x) # compare by number of elements
        elif issubclass(self.obj.datatype, complex):
            return lambda x: abs(x) # compares large numbers by integer orders of magnitude
        elif self.obj.key:
            return self.obj.key
        else:
            raise KeyInvalidError(f"Error: Must provide a key in order to compare min and max effectively!")

    def validate_is_comparator(self, element):
        """validates the stack element. ensures that it has comparison functionality (__lt__ & __gt__ & __eq__ & __ne__) - used to compare min, max and average items. """
        if not (hasattr(element, "__lt__") and hasattr(element, "__gt__")):
            raise KeyInvalidError(f"Key function must return an orderable type (supports < and >).")
        return element

    def validate_average_value(self, key_function, element):
        """
        Type Narrowing for element types - numeric = numeric total, non-numeric - len(item)
        Validation / type narrowing should be done on the key output, not the element itself:
        """
        # applies the key to the element
        transformed_element = key_function(element)

        if isinstance(transformed_element, (int, float, complex, bool)):
            result = transformed_element
        elif isinstance(transformed_element, (str, tuple)):
            result = len(transformed_element)
        else:
            try:
                result = len(transformed_element)
            except:
                raise KeyInvalidError(f"Error: Average Stack requires all Non Numeric values to implement __len__ for effective comparison.")
        return result

    def validate_key_consistency(self, key_function, element):
        """Ensures that the key function produces a consistent output type across all pushes."""

        transformed_element = key_function(element)

        if self.obj.is_empty():
            self.obj._key_type = type(transformed_element) 
        else:
            if type(transformed_element) is not self.obj._key_type:
                raise KeyInvalidError(f"The Custom Key input has flaky / unexpected expression type modification. Expected: {self.obj._key_type.__name__}, Got: {type(transformed_element)}")
    # endregion
