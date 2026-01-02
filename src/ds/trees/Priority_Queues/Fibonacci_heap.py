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
)
from abc import ABC, ABCMeta, abstractmethod
from array import array
import numpy
import ctypes
import random
from collections.abc import Sequence

# endregion


# region custom imports
from utils.validation_utils import DsValidation
from utils.representations import FibonacciHeapRepr
from utils.helpers import RandomClass
from utils.exceptions import *
from utils.constants import MIN_PQUEUE_CAPACITY

from adts.collection_adt import CollectionADT
from adts.priority_queue_adt import (
    MinPriorityQueueADT,
    MaxPriorityQueueADT,
    PriorityQueueADT,
)

from user_defined_types.generic_types import T, K, ValidDatatype, TypeSafeElement
from user_defined_types.key_types import iKey, Key

from ds.primitives.arrays.dynamic_array import VectorArray, VectorView
from ds.trees.Priority_Queues.priority_queue_utils import PriorityQueueUtils
from ds.trees.Priority_Queues.priority_entry import PriorityEntry

# endregion


class FibonacciHeap(PriorityQueueADT[T, K], CollectionADT[T], Generic[T, K]):
    """
    fibonacci heap data structure implementation:
    """
    def __init__(self, datatype: type, min_heap: bool = True) -> None:

        self._datatype = datatype
        self._pqueue_keytype: None | type = None
        self._min_heap = min_heap

        # composed objects
        self._utils = PriorityQueueUtils(self)
        self._validators = DsValidation()
        self._desc = FibonacciHeapRepr(self)

    @property
    def datatype(self):
        return self._datatype

    # ----- Meta Collection ADT Operations -----
    def __len__(self) -> int:
        return super().__len__()

    def __contains__(self, value: T) -> bool:
        return super().__contains__(value)

    def is_empty(self) -> bool:
        return super().is_empty()

    def clear(self) -> None:
        return super().clear()

    def __iter__(self):
        return

    # ------------ Utilities ------------

    def __str__(self) -> str:
        return self._desc.str_fibonacci_heap()

    def __repr__(self) -> str:
        return self._desc.repr_fibonacci_heap()

    # ----- Canonical ADT Operations -----
    # ----- Accessor ADT Operations -----
    def find_extreme(self) -> T:
        """
        """

    # ----- Mutator ADT Operations -----

    def insert(self, element: T, priority: K) -> None:
        """Create a new singleton tree & Add to root list"""


    def extract_extreme(self) -> T:
        """
        Delete -- meld its children into root list; update min.
        Consolidate trees so that no two roots have same rank.
        """

    def change_priority(self, element: T, priority: K) -> None:
        """
        If heap-order is not violated, just decrease the key of x.
        Otherwise, cut tree rooted at x and meld into root list.
        To keep trees flat: as soon as a node has its second child cut, cut it off and meld into root list (and unmark it)
        """
