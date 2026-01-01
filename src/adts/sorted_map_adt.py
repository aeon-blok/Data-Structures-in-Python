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
    Tuple,
    Iterable,
    TYPE_CHECKING,
)
from abc import ABC, ABCMeta, abstractmethod
from array import array
import numpy
import ctypes
import secrets
import math
import random
import time
from pprint import pprint


# region custom imports
from user_defined_types.generic_types import T, K, iKey
from adts.map_adt import MapADT

if TYPE_CHECKING:
    from adts.collection_adt import CollectionADT
    from ds.primitives.arrays.dynamic_array import VectorArray


# endregion



"""
Sorted Map ADT:
This includes all the operations of a Map ADT with some additional operations that relate to ordered items.
Sorted Map's are incredibly useful - and can be used to display automatically sorted information with no human intervention.
Maxima Sets are a great use case for Sorted Maps.
"""

class SortedMapADT(MapADT[T, K]):

    # ----- Canonical ADT Operations -----
    # ----- Accessors -----
    @abstractmethod
    def find_min(self) -> Optional[tuple]:
        """returns the entry with the smallest key. Sometimes this is known as firstkey()"""
        ...

    @abstractmethod
    def find_max(self) -> Optional[tuple]:
        """returns the entry with the largest key in the sorted map. Sometimes this is called lastkey()"""
        ...

    @abstractmethod
    def find_floor(self, key) -> Optional[tuple]:
        """returns the largest key that is smaller or equal to the specified key. Sometimes known as headmap()"""
        ...
    
    @abstractmethod
    def find_ceiling(self, key) -> Optional[tuple]:
        """returns the smallest key greater than or eqaul to the specified input key. Sometimes known as tailmap()"""
        ...

    @abstractmethod
    def predecessor(self, key) -> Optional[Tuple]:
        """returns the greatest key smaller than the specified key. Standard predecessor logic"""
        ...

    @abstractmethod
    def successor(self, key) -> Optional[Tuple]:
        """returns the smallest key greater than the specified key. Standard successor logic"""
        ...

    @abstractmethod
    def submap(self, start, stop) -> 'SortedMapADT[T, K]':
        """returns a new sorted map - that begins from the start key and ends at the stop key. this does not modify the original sorted map."""
        ...

    @abstractmethod
    def rank(self, key) -> int:
        """returns the number of keys strictly less than the specified key."""
        ...

    @abstractmethod
    def entries(self) -> Iterable[T]:
        """returns an array of the entries in the sorted map."""
        ...



    

