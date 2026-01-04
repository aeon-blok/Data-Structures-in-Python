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
import sys

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


def rod_cutting_naive_recursion(rod_length: int, lengths: list[int], prices: list[int]):
    """
    rod cutting problem - Naive recursive solution - follows CLRS guide.... 
    WARNING: This method is really slow - rod lengths over 40 might take 1hr....
    """
    # State = rod length

    # * base case
    if rod_length == 0:
        return 0
    # init as sentinel
    profit = -sys.maxsize
    # loop through the prices array.
    for idx, cut in enumerate(lengths):
        # ensure cut is less than the total rod length.
        if cut <= rod_length:
            # get the max between the current max and reductive step.
            profit = max(profit, prices[idx] + rod_cutting_naive_recursion(rod_length-cut,lengths, prices))
    # after recursive steps finish return the final profit.
    return profit


def main():
    fib_series = find_fibonacci_series_up_to(352)
    print(f"fibonacci series = {fib_series}")

    # print(f"Rod Cutting Problem:")
    # rod_length_prices = [1,5,8,9,10,17,17,20]
    # rod_lengths = [1,2,3,4,5,6,7,8]
    # target_rod_size = 5
    # print(f"Rod Cutting Problem: Cut the Target rod {target_rod_size}m into any number of pieces.")
    # print(f"Our Task is to get the most profit from our rod")
    # print(f"The prices for each specific length are \n{rod_lengths}\n{rod_length_prices}")
    # print(f"Constraints: The problem is unbounded, you can use unlimited cuts of the same length (e.g. 1+1+1+1)")
    # print(f"The order of the pieces dont matter...")
    # print(f"The Optimal Profit we can get is: ${rod_cutting_naive_recursion(target_rod_size,rod_lengths, rod_length_prices)}")


if __name__ == "__main__":
    main()
