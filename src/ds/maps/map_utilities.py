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
    TYPE_CHECKING,
    Literal,
)
from abc import ABC, ABCMeta, abstractmethod
from array import array
import numpy
import ctypes
import random
from collections.abc import Sequence
import math

# endregion


# region custom imports
from utils.custom_types import T, K, Key
from utils.validation_utils import DsValidation
from utils.exceptions import *
from utils.helpers import Ansi

if TYPE_CHECKING:
    from adts.collection_adt import CollectionADT
    from adts.map_adt import MapADT
    from adts.sequence_adt import SequenceADT

from ds.primitives.arrays.dynamic_array import VectorArray
from ds.maps.hash_functions import *
from ds.maps.probing_functions import *

# endregion


class MapUtils:
    """A collection of Utilities for Map Data Structures (hash tables, sets etc)"""
    def __init__(self, map_obj) -> None:
        self.obj = map_obj


    # -------------------------------- Utilities   --------------------------------
    def _is_prime_number(self, number):
        """Boolean Check if number is a prime."""
        if number < 2:
            return False
        for i in range(2, int(math.isqrt(number)) + 1):
            if number % i == 0:
                return False
        return True

    def find_next_prime_number(self, table_capacity):
        """Finds the next prime number larger than the current table capacity."""
        candidate = table_capacity + 1
        while True:
            if self._is_prime_number(candidate):
                return candidate
            candidate += 1

    # -------------------------------- Table Rehashing   --------------------------------

    def calculate_load_factor(self, total_elements, table_capacity) -> float:
        """calculates the load factor of the current hashtable"""
        return total_elements / table_capacity



