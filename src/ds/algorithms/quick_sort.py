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

# endregion


class QuickSort:
    """
    QuickSort Implementation: Quick Sort recursively sorts an array by:
    Selecting a pivot,
    Moving smaller values to the left and larger values to the right (partition),
    Recursively applying the same process to both sides
    Lomuto / Hoare might have issues with duplicates; Three-Way Partition handles them efficiently:
    """
    def __init__(self, partition: Literal['lomuto', 'hoare', '3 way'] = '3 way') -> None:
        self.partition = partition
    # --------------- Partition Operations ---------------

    def _lomuto_partition(self, input_array, min_index: Index, max_index: Index) -> Index:
        """
        Each partition permanently fixes one element in its final sorted position. The algorithm never sorts the whole array repeatedly — it works on subsegments.
        Pivot chosen as the last element.
        """
        pivot = input_array[max_index]
        i = min_index - 1  # boundary for elements < pivot
        # * search the array and perform comparisions and swaps....
        for j in range(min_index, max_index):
            # compare element to pivot and swap if element is smaller.
            if input_array[j] < pivot:
                # extends the "smaller" subsector by increasing the index of i (which is the boundary)
                i += 1
                # * swap elements i & j
                input_array[i], input_array[j] = input_array[j], input_array[i]

        # place pivot in its final position (swap) -- its moved from smaller elements to larger elements...
        input_array[i + 1], input_array[max_index] = input_array[max_index], input_array[i + 1]
        return i + 1

    def _hoare_partition(self, input_array, min_index: Index, max_index: Index):
        """
        Pivot is usually first element
        Two pointers: i (left → right), j (right → left)
        Swap elements when arr[i] > pivot and arr[j] < pivot
        Stop when i >= j
        Pivot may not end in its final position
        Returns partition boundary (j), not pivot index
        """
        pivot = input_array[min_index]
        i = min_index - 1
        j = max_index + 1

        while True:
            # Move i right until arr[i] >= pivot
            i += 1
            while input_array[i] < pivot:
                i += 1

            # Move j left until arr[j] <= pivot
            j -= 1
            while input_array[j] > pivot:
                j -= 1

            if i >= j:
                return j  # partition boundary (not pivot index)

            # Swap out-of-place elements
            input_array[i], input_array[j] = input_array[j], input_array[i]

    def _three_way_partition(self, input_array, min_index: Index, max_index: Index):
        """Also called the Dutch Flag Partition. Splits into 3 regions (less than, equal and greater than)"""
        # choose first element as pivot
        pivot = input_array[min_index]  # first element.
        lt = min_index  # end of less-than region
        i = min_index + 1  # current element
        gt = max_index  # start of greater-than region

        while i <= gt:
            if input_array[i] < pivot:
                input_array[lt], input_array[i] = input_array[i], input_array[lt]
                lt += 1
                i += 1
            elif input_array[i] > pivot:
                input_array[i], input_array[gt] = input_array[gt], input_array[i]
                gt -= 1
            else:  # input_array[i] == pivot
                i += 1
        return lt, gt

    # --------------- Sort Operation ---------------

    def _lomuto_sort(self, input_array, min_index: Index, max_index: Index) -> None:
        """Sorts array via lomuto partition method"""

        # base case: arrays - of 0 and 1 size are already sorted.
        if min_index < max_index:
            part = self._lomuto_partition(input_array, min_index, max_index)
            # both sides of the partition are sorted independently
            self._lomuto_sort(input_array, min_index, part - 1)
            self._lomuto_sort(input_array, part + 1, max_index)

    def _hoare_sort(self, input_array, min_index: Index, max_index: Index) -> None:
        """Sorts array with Hoare Partition (fewer swaps and better performance on large arrays than the lomuto partition)"""
        if min_index < max_index:
            part = self._hoare_partition(input_array, min_index, max_index)
            self._hoare_sort(input_array, min_index, part)
            self._hoare_sort(input_array, part + 1, max_index)

    def _three_way_sort(self, input_array, min_index: Index, max_index: Index) -> None:
        """
        Sorts array via three way partition (great for arrays with many duplicates.)
        reduces recursion depth and swaps.
        """
        if min_index >= max_index: 
            return

        lt, gt = self._three_way_partition(input_array, min_index, max_index)
        # Recursively sort the < pivot and > pivot regions
        self._three_way_sort(input_array, min_index, lt - 1)  # < pivot
        self._three_way_sort(input_array, gt + 1, max_index)  # > pivot

    # --------------- Strategy Pattern  ---------------

    def sort(self, input_array):
        """Can choose which type of partition you wish to use. (lomuto, hoare, 3 way)"""
        min_index = 0
        max_index = len(input_array) - 1

        if min_index >= max_index:
            return input_array
        elif self.partition == 'lomuto':
            self._lomuto_sort(input_array, min_index, max_index)
            return input_array
        elif self.partition == 'hoare':
            self._hoare_sort(input_array, min_index, max_index)
            return input_array
        elif self.partition == '3 way':
            self._three_way_sort(input_array, min_index, max_index)
            return input_array
        else:
            raise DsTypeError(f"Error: Partition method must be a valid partition type (lomuto, hoare, 3 way)")


# ------------------------------- Main: Client Facing Code: -------------------------------
def main():

    test_array = [3245,543,765,86797,89786,780,68467,57,6,345,345,32,534,25,24547,68,769,86,34235,3,1,1,5246,34,7546,8675,76,895,6643,15,34,654,867,987,97,8,32,362,4567,4568,9,67,976,743]
    print(f"\nTesting Lomuto Partition Quick Sort")
    lomuto = QuickSort(partition='lomuto')
    print(test_array)
    result = lomuto.sort(test_array)
    print(result)

    test_array_2 = [3245,543,765,86797,89786,780,68467,57,6,345,345,32,534,25,24547,68,769,86,34235,3,1,1,5246,34,7546,8675,76,895,6643,15,34,654,867,987,97,8,32,362,4567,4568,9,67,976,743]
    print(f"\nTesting Hoare Partition Quick Sort")
    hoare = QuickSort(partition='hoare')
    print(test_array_2)
    result = hoare.sort(test_array_2)
    print(result)

    test_array_3 = [3245,543,765,86797,89786,780,68467,57,6,345,345,32,534,25,24547,68,769,86,34235,3,1,1,5246,34,7546,8675,76,895,6643,15,34,654,867,987,97,8,32,362,4567,4568,9,67,976,743]
    print(f"\nTesting 3 way Partition Quick Sort")
    three_way = QuickSort(partition='3 way')
    print(test_array_3)
    result = three_way.sort(test_array_3)
    print(result)

    print(f"\nTesting Empty Array")
    test_array_empty = []
    print(test_array_empty)
    qs = QuickSort().sort(test_array_empty)
    print(qs)

    print(f"\ntest 1 element array")
    test_array_solo = [1]
    print(test_array_solo)
    quick = QuickSort().sort(test_array_solo)
    print(quick)

    print(f"\nTesting already sorted array")
    


if __name__ == "__main__":
    main()
