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

# endregion


def insertion_sort(input_array):
    """
    Insertion Sort Algorithm: 
    """
    # starts at 1 because 0 is considered sorted by default.
    for i in range(1, len(input_array)):
        key = input_array[i]
        sorted = i-1
        # * moves backwards through the sorted elements. 
        # elements that are greater than the key - move one position ahead
        while sorted >= 0 and input_array[sorted] > key:
            input_array[sorted + 1] = input_array[sorted]
            # * moves the pointer backwards to check the next element.
            sorted -= 1
        # * correct position found: insert element after the last sorted element in the sorted subarray.
        input_array[sorted + 1] = key
    return input_array


def bubble_sort(input_array):
    """
    Bubble Sort Algorithm: Each pass ensures that the largest unsorted element “bubbles” to its correct position at the end.
    After the first pass, the last element is guaranteed to be the largest. After the second pass, the second-to-last element is in place, and so on.
    """
    array_length = len(input_array)
    for i in range(array_length-1):
        swapped = False # early exit optimization
        # * iterates over the unsorted part of the array. (array length - i - 1) -- this ensures we dont transform already sorted elments.
        for j in range(0, array_length-i-1):
            # * checks 2 adjacent elements.
            if input_array[j] > input_array[j+1]:
                # * swaps the elements
                input_array[j], input_array[j+1] = input_array[j+1], input_array[j]
                swapped = True
        # * Exit Condition: if no elements swapped
        if not swapped: break
    # Returns the sorted array in ascending order.
    return input_array

def selection_sort(input_array):
    """
    Repeatedly select the minimum element from the unsorted portion and swap it into its final position at the front. After each pass, the sorted region grows by one.
    Selection sort trades extra comparisons for minimal swaps. This swap can reorder equal elements. Therefore, selection sort is not stable by default.
    At the start of iteration i, the subarray input_array[0:i] is sorted and contains the i smallest elements in correct order.
    """
    array_length = len(input_array)
    # i: represents the boundary of the sorted elements in the array.
    for i in range(array_length-1):
        # min index: Stores the index of the smallest value found during the inner scan.
        min_index = i
        # * Scans the unsorted portion to identify the smallest element
        for search_index in range(i + 1, array_length):
            if input_array[search_index] < input_array[min_index]:
                min_index = search_index    # update smallest element to found value
        # * swap elements -- if the smallest element is not in the sorted elements - swap with 
        if min_index != i:
            input_array[i], input_array[min_index] = input_array[min_index], input_array[i]
    return input_array


# ------------------------------- Main: Client Facing Code: -------------------------------

def main():
    print(f"\nTesting Insertion Sort:")
    test_array = [2342,64,6547,56,877876,98534646,654,5324,521,213,52,12312,1,343,6,756867,99,78,453,523,432]
    print(test_array)
    result = insertion_sort(test_array)

    print(result)

    print(f"\nTesting Bubble Sort:")
    test_array_2 = [34232,543,654,765,978989,86,634,535,25,765,78,76,89,35,3,24,1,4,65568,569,870,7,56734,563,47,5]
    print(test_array_2)
    bubble = bubble_sort(test_array_2)
    print(bubble)


    print(f"\nTesting Selection Sort:")
    test_array_3 = [8543,432,5,68,78,67,25,34,8,969,6543,61,457564,8675,978,89,98,645,4,6354,86,579,68976,6543,254,35,345,345,7546,7,435,432,46,2,1,34,53568,675,9887,9,354,63,25,234,645,765,845,78,5,987,60897,90,]
    print(test_array_3)
    selection = selection_sort(test_array_3)
    print(selection)

if __name__ == "__main__":
    main()
