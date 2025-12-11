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


def merge_sort(input_array, start=0, end=None):
    """
    Merge Sort Algorithm: in-place variant.
    recursively splits an array in half until each sub array has only one element. 
    then merges all the sub arrays back together.
    """

    def merge_halves(input_array, start, mid, end):
        """This function combines the left and right halves into a sorted segment."""
        i = start   # current element in left half
        j = mid + 1 # current element in right half

        # while we havent reached the end of either half.
        while i <= mid and j <= end:
            # * if the left half element is smaller or equal to the right half element - move to the next element.
            if input_array[i] <= input_array[j]:
                i += 1
            # * otherwise we must swap the elements - the right element moves into the left half.
            else:
                # * stores the element from the right half that needs to be moved to the left.
                value = input_array[j]
                # * move backwards through array and Shift all elements from i to j-1 one step to the right
                for k in range(j, i, -1):
                    input_array[k] = input_array[k - 1]
                input_array[i] = value  # move element to the left half.
                # update ponters
                i += 1
                mid += 1  # mid moves because we inserted before mid
                j += 1  

    # size of array (end pointer)
    if end is None: end = len(input_array)-1
    # * base case - stops recursion when there is only 1 element in the array. (this element is sorted.)
    if start >= end: return
    # derive middle index
    mid = (start + end) // 2
    # * split the array into halves. recursively sort these also. (mutator - no return value)
    merge_sort(input_array, start, mid) # left half
    merge_sort(input_array, mid+1, end) # right half
    # * sort & combine the array halves.
    merge_halves(input_array, start, mid, end)
    return input_array


# ------------------------------- Main: Client Facing Code: -------------------------------
def main():

    test_array = [3245,543,765,86797,89786,780,68467,57,6,345,345,32,534,25,24547,68,769,86,34235,3,1,1,5246,34,7546,8675,76,895,6643,15,34,654,867,987,97,8,32,362,4567,4568,9,67,976,743]
    print(test_array)
    merge = merge_sort(test_array)
    print(merge)


if __name__ == "__main__":
    main()
