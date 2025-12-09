# region standard imports

from typing import (
    Generic,
    TypeVar,
    Dict,
    Optional,
    Callable,
    Any,
    cast,
    Iterator,
    Generator,
    Iterable,
    TYPE_CHECKING,
    NewType,
    List,
)
from abc import ABC, ABCMeta, abstractmethod
from array import array
import numpy
import ctypes
import random
import math

from collections.abc import Sequence

# endregion


# region custom imports
from user_defined_types.generic_types import T, Index
from user_defined_types.hashtable_types import NormalizedFloat

from utils.validation_utils import DsValidation
from utils.exceptions import *

if TYPE_CHECKING:
    from adts.collection_adt import CollectionADT
    from adts.sequence_adt import SequenceADT


class LinearSearch:
    """Linear Search Algorithms and variants... -- O(N)"""
    def __init__(self) -> None:
        pass

    def sentinel_linear_search(self, input_array, target_value) -> Optional[Index]:
        """
        Optimization of linear search: put the target at the end as a “sentinel” to avoid extra comparisons. -- O(N)
        Does NOT require Sorted array.
        It’s just a linear scan with a sentinel to avoid one comparison per iteration.
        """
        array_length = input_array.size
        array_copy = input_array.copy()
        # add target to the end of array. (sentinel.)
        array_copy.append(target_value)
        # idx tracker
        idx = 0
        # loop through array items.
        while array_copy[idx] != target_value:
            idx+= 1
        if idx < array_length:
            return idx
        return None

    def jump_search(self, sorted_array, target_value):
        """
        Only works on sorted arrays. Designed for sorted arrays stored in sequential-access memory (like linked lists or disks where jumping around is expensive).
        You jump in blocks (√n steps), then scan linearly within the block.
        """

        array_length = sorted_array.size
        step = int(math.sqrt(array_length))
        prev = 0

        # Jump in blocks until target could be in current block
        while prev < array_length and sorted_array[min(prev + step - 1, array_length - 1)] < target_value:
            prev += step

        # linear search when target value is in block.
        for i in range(prev, min(prev + step, array_length)):
            if sorted_array[i] == target_value:
                return i

        return None





