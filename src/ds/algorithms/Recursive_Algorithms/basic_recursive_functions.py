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
from user_defined_types.generic_types import T, Index, PositiveNumber
from user_defined_types.hashtable_types import NormalizedFloat

from utils.validation_utils import DsValidation
from utils.exceptions import *
# endregion


# todo recursive function for solving palindromes.

def find_sum(start: int, stop: int) -> int:
    """
    recursively sums the numbers in the specified range.
    works with negative numbers.
    """

    # * type check
    if not isinstance(start, int):
        raise TypeError(f"Error: Input must be an integer.")
    if not isinstance(stop, int):
        raise TypeError(f"Error: Input must be an integer.")

    # * exit condition
    if start > stop:
        return 0

    # * Base Case -- nothing left to sum.
    if start == stop:
        return start

    return start + find_sum(start + 1, stop)

def find_factorial(n: int) -> int:
    """
    Factorial counts how many different ordered arrangements exist for n distinct things.
    this function recursively calcuates the factorial for a positive integer
    """
    n = PositiveNumber(n)

    # * base case (0 or 1)
    if n <= 1:
        return 1
    
    return n * find_factorial(n-1)

def find_fibonacci_series_up_to(n: int):
    """finds the fibonacci series of specified number"""

    n = PositiveNumber(n)

    def recurse(remain, current, next, result):

        # * base case
        if current > remain:
            return result
        
        result.append(current)
        return recurse(remain - current, next, current+next, result)
    
    return recurse(n, 1, 1, [])

def is_word_palindrome(input: str) -> bool:
    """"""
    pass


def main():
    fib_series = find_fibonacci_series_up_to(352)
    print(f"fibonacci series = {fib_series}")


if __name__ == "__main__":
    main()


