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
    List
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


class BinarySearch():
    """a collection of binary search algorithms that are utilized specifically for sorted arrays."""
    def __init__(self, noise_probability=0.1, error_tolerance=0.1,) -> None:
        self._noise_probability = NormalizedFloat(noise_probability)
        self._error_tolerance = NormalizedFloat(error_tolerance)
    
    # --------------- Utility Operations ---------------

    def __repr__(self) -> str:
        return f"[{self.__class__.__qualname__}: {hex(id(self))}]"
    
    def check_sorted_array_exists(self, sorted_array):
        """Ensures the sorted array input is not None"""
        if sorted_array is None:
            raise DsInputValueError(f"Error: Sorted Array Input Cannot be None. Please input a valid sorted array.")
        if len(sorted_array) ==0:
            raise DsInputValueError(f"Error: Sorted Array must have more than 0 elements. (for binary search to work correctly.)")
        return sorted_array

    def check_target_exists(self, target_value):
        """ensures the target value is not None"""
        if target_value is None:
            raise DsInputValueError(f"Error: Target Value cannot be None.")
        return target_value
  
    # --------------- Canonical Operations ---------------
    # --------------- Lookup Search Operations ---------------
    def classic_binary_search(self, target_value, sorted_array) -> Optional[Index]:
        """
        Binary Search Algorithm using iterative style
        returns target value index.
        Only works with Sorted Arrays.
        """

        # * validate inputs.
        # we also need to check that the target value and the input array are the same type.
        # and that they key objects are also the same type.
        sorted_array = self.check_sorted_array_exists(sorted_array)
        target_value = self.check_target_exists(target_value)
        
        # * Empty Sorted Array Guard Clause:
        if len(sorted_array) == 0: return 0

        left = 0
        right = len(sorted_array) - 1


        # * while an index exists that could still contain the target value - search
        while left <= right:
            # * find the midpoint index (this formula avoids integer overflow and stalling)
            middle = left + (right - left) // 2  
            # * Compare Target Value with Mid Value
            if sorted_array[middle] == target_value:
                return middle
            # move min to middle (+1 because middle is already checked and does not contain the value.)
            elif sorted_array[middle] < target_value:
                left = middle + 1
            # move max to middle (-1)
            else:
                right = middle - 1
        # element not present in array.
        return None

    def recursive_binary_search(self, target_value, sorted_array) -> Optional[Index]:
        """
        Binary Search Algorithm - recursive style. The recursive function calls shrink the search space by half
        returns the target value - index.
        """

        # validate inputs
        sorted_array = self.check_sorted_array_exists(sorted_array)
        target_value = self.check_target_exists(target_value)
        
        # * Empty Sorted Array Guard Clause:
        if len(sorted_array) == 0: return 0
    
        left, right = 0, len(sorted_array) - 1

        def _recursive_search(sorted_array, target_value, left, right):
            """recursive helper method for recursive binary search"""
            # * while an index exists that could still contain the target value - search
            if right >= left:
                # * find the midpoint index
                mid_index = left + (right - left) // 2
                # * comparisons
                if sorted_array[mid_index] == target_value:
                    return mid_index
                # move upper bound to middle index and recursively search
                elif sorted_array[mid_index] > target_value:
                    return _recursive_search(sorted_array, target_value, left, mid_index-1)
                # move lower bound to middle index and recursively search
                else:
                    return _recursive_search(sorted_array, target_value, mid_index+1, right)
            # element not present.
            else:
                return None

        return _recursive_search(sorted_array, target_value, left, right)
    
    def binary_exponential_search(self, target_value, sorted_array) -> Optional[Index]:
        """
        Exponential Search works on Unbounded sequences (close to infinite etc...)
        bound increases exponentially until it hits either: the end of the array, or is greater than the target value.
        """
        sorted_array = self.check_sorted_array_exists(sorted_array)
        target_value = self.check_target_exists(target_value)
        
        # * Empty Sorted Array Guard Clause:
        if len(sorted_array) == 0: return 0

        # initialize variables
        bound = 1
        array_length = len(sorted_array) - 1

        # * exponentially increase the bound via looping until we reach the end of the array, or bound is >= target
        while bound < array_length and sorted_array[bound] < target_value:
            bound *= 2  # exponentially double

        # * define search range
        left = bound // 2
        right = min(bound, array_length - 1)

        # * binary search
        while left <= right:
            mid = left + (right - left) // 2

            if sorted_array[mid] == target_value:
                return mid
            elif sorted_array[mid] < target_value:
                left = mid + 1
            else:
                right = mid - 1
        return None

    def binary_interpolation_search(self, target_value, sorted_array) -> Optional[Index]:
        """Binary search with an advanced technique to calculate the 'midpoint' index closer to the target value than a standard binary search."""
        # validate inputs
        sorted_array = self.check_sorted_array_exists(sorted_array)
        target_value = self.check_target_exists(target_value)
        
        # * Empty Sorted Array Guard Clause:
        if len(sorted_array) == 0: 
            return 0

        left, right = 0, len(sorted_array) - 1

        while left <= right:
            # safeguard for small ranges.
            if right - left < 4:
                # right + 1 - python slicing does not count the end of an array so you have to add 1
                sliced_index_result = self.classic_binary_search(target_value, sorted_array[left:right+1])
                if sliced_index_result is not None:
                    return left + sliced_index_result
                else: return None

            # prevent division by zero:
            if sorted_array[left] == sorted_array[right]:
                if sorted_array[left] == target_value: return left
                else: return None

            # uses interpolation formula to fund midpoint that is closer to the target value.
            mid = left + ((target_value - sorted_array[left]) * (right - left)) // (sorted_array[right] - sorted_array[left])

            # saftey for erroneous mid values.
            if mid < left or mid > right:
                return None
            elif sorted_array[mid] == target_value:
                return mid
            elif sorted_array[mid] < target_value:
                left = mid + 1
            else:
                right = mid - 1
        return None

    def noisy_binary_search(self, target_value, sorted_array, simulated_noise=True):
        """
        Noisy Binary Search: Used when you have a noisy and unreliable input value (from a sensor etc...)
        Each comparison is repeated a specific user defined number of times for a majority vote on the correct response.
        noise_probability = This is the probability that a single comparison is wrong. (how unreliable the oracle is.)
        Has the option to simulate noisy comparisons - (in real life memory corruption etc...)
        """
        # validate inputs...
        error_tolerance = self._error_tolerance # delta
        noise_probability = self._noise_probability # predicted probability that a single comparison is incorrect
        sorted_array = self.check_sorted_array_exists(sorted_array)
        target_value = self.check_target_exists(target_value)
        
        # * Empty Sorted Array Guard Clause:
        if len(sorted_array) == 0: return 0

        array_length = len(sorted_array) - 1
        binary_steps = math.ceil(math.log2(array_length))   # number of steps taken to arrive at the end result. (via binary search)
        # delta = the overall risk of the whole search failing, which you want to keep small.

        # epsilon = Per-step failure probability
        epsilon =  error_tolerance / binary_steps
        # repeats is math calcuated formula to ensure error tolerance rate is conformed to.
        repeats = math.ceil(math.log(2 / epsilon) / (2 * (0.5 - noise_probability)**2))

        # initialize search vars
        left, right = 0, array_length - 1

        # defines search range.
        while left <= right:
            # find midpoint.
            mid = left + (right - left) // 2

            # * comparisons repeated a user specified number of times. - every time a counter is updated with the result.
            # init counters -- under < target, equal == target, over > target
            under = equal = over = 0

            for _ in range(repeats):
                # * determine comparison result:
                if sorted_array[mid] < target_value:
                    result = -1
                elif sorted_array[mid] > target_value:
                    result = 1
                else:
                    result = 0
                # * simulate noisy comparison if enabled
                if simulated_noise and result != 0 and random.random() < noise_probability:
                    # flip the comparison randomly.
                    result *= -1

                if result == -1:
                    under += 1
                elif result == 1:
                    over += 1
                else:
                    equal += 1

            # * noisy majority vote:
            if equal >= under and equal >= over:
                return mid
            elif under >= over:
                left = mid + 1
            else:
                right = mid - 1

        # element not found
        return None

    # --------------- Boundary Search Operations ---------------
    def binary_search_lower_bounds(self, target_value, sorted_array) -> Index:
        """
        finds the first element that is equal to or larger than the target value. then returns the index of this element.
        the key difference from a regular binary search — we don’t stop when we find x; we keep narrowing to the first index ≥ x.
        the lower bound can be the successor if the target is not in the array.
        """

        # validate inputs
        sorted_array = self.check_sorted_array_exists(sorted_array)
        target_value = self.check_target_exists(target_value)
        
        # * Empty Sorted Array Guard Clause:
        if len(sorted_array) == 0: return 0

        left, right = 0, len(sorted_array)
        # keep searching as long as the range has indices
        while left < right:
            # * find middle index (divide & conquer)
            mid = left + (right - left) // 2
            # * comparisons
            if sorted_array[mid] < target_value:
                left = mid + 1
            else:
                right = mid
        return left

    def binary_search_upper_bounds(self, target_value, sorted_array) -> Index:
        """
        finds the first element that is strictly Larger than the target value and returns the index of this element.
        the upper bound result is always the successor
        """
        sorted_array = self.check_sorted_array_exists(sorted_array)
        target_value = self.check_target_exists(target_value)
        
        # * Empty Sorted Array Guard Clause:
        if len(sorted_array) == 0: return 0

        left, right = 0, len(sorted_array)
        while left < right:
            mid = left + (right - left) // 2
            if sorted_array[mid] <= target_value:  # key difference: <= instead of <
                left = mid + 1
            else:
                right = mid
        return left

    def binary_search_predecessor(self, target_value, sorted_array) -> Optional[Index]:
        """this returns the predecessor of the target value. That is: largest element in the array < x"""

        sorted_array = self.check_sorted_array_exists(sorted_array)
        target_value = self.check_target_exists(target_value)
        
        # * Empty Sorted Array Guard Clause:
        if len(sorted_array) == 0: return 0

        lower_bounds: Index = self.binary_search_lower_bounds(target_value, sorted_array) 
        # element just before lower bound
        if lower_bounds != 0:
            return lower_bounds - 1
        else: 
            return None  # no element less than target_value

    def binary_search_successor(self, target_value, sorted_array) -> Optional[Index]:
        """returns the successor -- the smallest element > x"""
        sorted_array = self.check_sorted_array_exists(sorted_array)
        target_value = self.check_target_exists(target_value)
        
        # * Empty Sorted Array Guard Clause:
        if len(sorted_array) == 0: return 0

        upper_bounds: Index = self.binary_search_upper_bounds(target_value, sorted_array)
        if upper_bounds < len(sorted_array):
            return upper_bounds
        else:
            return None  # no element greater than target_value

    def binary_search_rank(self, target_value, sorted_array) -> Index:
        """returns the index where the target would go in a sorted array. Also the index is equal to the number of elements strictly less than the target."""
        sorted_array = self.check_sorted_array_exists(sorted_array)
        target_value = self.check_target_exists(target_value)
        
        # * Empty Sorted Array Guard Clause:
        if len(sorted_array) == 0: return 0

        left, right = 0, len(sorted_array)
        while left < right:
            mid = (left + right) // 2
            if sorted_array[mid] < target_value:
                left = mid + 1
            else:
                right = mid
        # this will be the number of elements lower than the target
        return left
























