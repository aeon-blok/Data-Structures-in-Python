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


class RadixSort:
    """radix sort algorithm implementation:"""
    def __init__(self) -> None:
        pass

    def _digit_at(self, num: int, exp: int):
        """
        Return digit at position 'exp' for arbitrarily large integers.
        returns the decimal digit of num at the place corresponding to its exponent level (001 = unit, 010 = tens, 100 = hundreds, 1000 = thousands... the digit returned in these instances is 1).
        """
        return (num // exp) % 10

    def _recursive_msd_helper(self, input_array, start: Index, end: Index, exp: int) -> None:
        """
        Recursively sorts a subarray of integers using Most Significant Digit (MSD) radix sort.
        Step 1: Counts the occurences of digits 0-9 at a specific exponent level (1000, 100, 10, 1) starting at the Most Significant Digit.
        Step 2: Prefix Sums: Computes the start index, and end index for the range of the specified subsection of the array.
        Step 3: Iterates through the subsection and swaps elements into their correct ordered place.
        Step 4: recursively applies the same function to an ever decreasing subsection of the array until there is only 1 element left (which is by definition sorted.)
        """

        # * base case: (Exit condition) - no more elements to sort.
        if end - start <= 1 or exp == 0:
            return

        # * 1.) counts the occurences of a digit for a specific exponent level.
        # Creates an array of size 10, one slot for each possible decimal digit 0..9.
        # count[d] will hold how many numbers in the current subarray have digit d at the current exp place.
        count = VectorArray(10, object)
        for _ in range(10): count.append(0)
        for i in range(start, end):
            # the value of the digit (0-9) at this current exponent.
            digit = self._digit_at(input_array[i], exp)
            count[digit] += 1   # increment count. (how many numbers have digit x(0-9) at the current exponent level.)

        # * 2.) compute start & end positions: Prefix Sums (turn bucket size into index in the array)
        # we are computing the range of index that these numbers will fall under.
        # initialize array with dummy values. (0)
        start_index_arr = VectorArray(10, object)
        for _ in range(10): start_index_arr.append(0)
        end_index_arr = VectorArray(10, object)
        for _ in range(10): end_index_arr.append(0)
        # This is the prefix sum step: converts bucket sizes → starting indices.
        for i in range(1,10):
            start_index_arr[i] = start_index_arr[i-1] + count[i-1]
        # shift start indices by starting point of subarray.
        for i in range(10):
            start_index_arr[i] += start
        # compute the end indices...
        for i in range(10):
            end_index_arr[i] = start_index_arr[i] + count[i]

        # * 3.) place elements into the output array in sorted order.
        i = start
        while i < end:
            # init vars
            digit = self._digit_at(input_array[i], exp)
            target_idx = start_index_arr[digit]
            sector_start = end_index_arr[digit] - count[digit]
            sector_end = end_index_arr[digit]

            # target index is not in bucket range.
            if not (sector_start <= target_idx < sector_end):
                i += 1
                continue
            # element is in correct position
            elif target_idx == i:
                # move to next index
                start_index_arr[digit] += 1
                i += 1
            # number not in the correct index range.
            else:
                # swap elements in the output array
                input_array[i], input_array[target_idx] = input_array[target_idx], input_array[i]
                # move the next free position forward
                start_index_arr[digit] += 1


        # * 4.) Move to the next exp level down - and recursively sort.
        # calculates the next exp level down (from hundreds to tens etc) for LSD it would go up
        next_exp = exp // 10
        for digit in range(10):
            # * prefix sums: compute the ranges of indices for this exponent level
            # Why not use start_index_arr[x]? During in-place placement, start_index_arr[x] has been incremented as numbers were swapped into place. we need the original start
            subsector_start = end_index_arr[digit] - count[digit]
            subsector_end = end_index_arr[digit]
            # * recursively apply the same method to the new subsection of the array and sort.
            if subsector_end - subsector_start > 1:
                self._recursive_msd_helper(input_array, subsector_start, subsector_end, next_exp)

    def inplace_msd_sort(self, input_array):
        """
        stable, inplace radix sort implementation.
        can handle negative integers
        mostly inplace.....
        """

        if not input_array: return input_array

        array_length = len(input_array)
        # * apply offset for negative values (allows negative integers in the array.)
        min_val = min(input_array)
        negative_offset = -min_val if min_val < 0 else 0
        if negative_offset:
            for i in range(len(input_array)):
                input_array[i] += negative_offset

        # find highest power of 10
        max_val = max(input_array)
        exp = 1

        # finds the exponent level of the highest digit (for MSD) (1, 10, 100, 1000, 10000)
        exp = 10 ** (len(str(max_val)) - 1)

        # * call helper method. this is the main functionality - recursively applies this method until there is only 1 element (which is sorted by definition)
        self._recursive_msd_helper(input_array, 0, array_length, exp)

        # * remove negative offset
        if negative_offset:
            for i in range(len(input_array)):
                input_array[i] -= negative_offset

        return input_array

    def lsd_sort(self, input_array):
        """uses Least Significant Digit"""
        pass


# ------------------------------- Main: Client Facing Code: -------------------------------

def main():

    test_array = [3245,543,765,86797,89786,780,68467,57,6,345,345,32,534,25,24547,68,769,86,-34235,3,-1,1,1,1,5246,34,7546,8675,76,895,6643,15,-34,654,867,987,97,8,32,362,4567,4568,9,67,976,743]
    print(f"\nTesting MSD radix sort.")
    print(test_array)
    radix = RadixSort().inplace_msd_sort(test_array)
    print(radix)

if __name__ == "__main__":
    main()
