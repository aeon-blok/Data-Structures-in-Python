# region imports

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
)
from abc import ABC, ABCMeta, abstractmethod
import numpy
import ctypes

# endregion

# arrays
ARRAY_GROWTH_FACTOR: int = 2    # amount to resize when growing array
ARRAY_SHRINK_FACTOR: int = 2    # amount to resize when shrinking array
SHRINK_CAPACITY_RATIO: int = 4  # divide capacity by this number   capacity // 4
ARRAY_MIN_CAPACITY: int = 4

CTYPES_DATATYPES = {
    int: ctypes.c_int,
    float: ctypes.c_double,
    bool: ctypes.c_bool,
    str: ctypes.py_object,  # arbitrary Python object (strings)
    object: ctypes.py_object,  # any Python object
}

NUMPY_DATATYPES = {
    int: numpy.int32,
    float: numpy.float64,
    bool: numpy.bool_,
}

# linked lists
SLL_SEPERATOR = " ->> "
DLL_SEPERATOR = " <-> "

# hashtables
MIN_HASHTABLE_CAPACITY: int = 10
DEFAULT_HASHTABLE_CAPACITY: int = 20
MAX_LOAD_FACTOR: float = 0.6
HASHTABLE_RESIZE_FACTOR: int = 2
BUCKET_CAPACITY: int = 10
COLLISIONS_THRESHOLD: float = 0.13
PROBES_THRESHOLD: float = 0.15
TOMBSTONES_THRESHOLD: float = 0.15
AVERAGE_PROBES_LIMIT: float = 4
TOMBSTONE_MARKER: str = "🪦"
LOAD_FACTOR_SYMBOL: str = "🏋️" or "🚚"
COLLISIONS_SYMBOL: str = "💥" or "⚠️"
REHASH_SYMBOL: str = "♻️" or "⚙️" or "🔧"
PROBE_SYMBOL: str = "🔍"
AVERAGE_PROBES_SYMBOL: str = "Avg 🔍"

# priority queues
MIN_PQUEUE_CAPACITY = 4

# trie
ALPHABET = [
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z"
]

# b trees
PAGE_SIZE: int = 4096
