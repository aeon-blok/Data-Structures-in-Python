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
    Literal,
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

from ds.primitives.arrays.dynamic_array import VectorArray

# endregion


class CountingSort:
    """Counting Sort Algorithm Implementation"""
    def __init__(self) -> None:
        pass

    def classic_sort(self, input_array):
        """Classic Counting Sort Implementation -- Only works on integers."""

        # * step 1: find the Max value
        min_val = min(input_array)
        max_val = max(input_array)
        # range_size guarantees all values from min_val to max_val map to valid indices.
        range_size = max_val - min_val + 1

        # init counter array.
        count = VectorArray(range_size, int)

        # * step 2: count frequencies
        for num in input_array:
            index = num - min_val  # maps the value to the correct count slot
            count[index] += 1   # increment the count for this value.

        # * step 3: Prefix Sums: Cumatively add the left index count to the target index (index 0 stays the same)
        for i in range(1, range_size):
            # This calculates the ending index in the sorted array for each value.
            # This step is crucial for stability, as it ensures elements appear in output in the same order as input for duplicates.
            count[i] += count[i-1]

        # * step 4: Add elements into the sorted output array (stable and in order.)
        sorted_array = VectorArray(len(input_array), int)
        # * prefill array with dummy values to allow setitem to work in random locations.
        for i in range(len(input_array)): sorted_array.append(0)
        # Reversing input ensures stability. (equal values are ordered same to insertion order.)
        for num in reversed(input_array):
            index = num - min_val   # derive index
            # * decrementing the elements count in the counter array
            count[index] -= 1   
            sorted_array[count[index]] = num

        return sorted_array


# ------------------------------- Main: Client Facing Code: -------------------------------

def main():

    print(f"\nTesting Classic Counting Sort (integers only -- negative numbers allowed)")
    test_array = [3245,543,765,86797,89786,780,68467,57,6,345,345,32,534,25,24547,68,769,86,34235,3,1,1,5246,34,7546,-8675,-76,-76,76,895,6643,15,34,654,867,987,97,8,32,362,4567,4568,9,67,976,743]
    print(test_array)
    cs = CountingSort().classic_sort(test_array)
    print(cs)

if __name__ == "__main__":
    main()
