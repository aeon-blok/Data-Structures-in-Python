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
from utils.representations import PQueueRepr
from utils.helpers import RandomClass
from utils.exceptions import *

from adts.collection_adt import CollectionADT
from adts.priority_queue_adt import MinPriorityQueueADT, MaxPriorityQueueADT, PriorityQueueADT
from adts.sequence_adt import SequenceADT


from ds.primitives.arrays.dynamic_array import VectorArray, VectorView
from ds.trees.Priority_Queues.priority_queue_utils import PriorityQueueUtils

from user_defined_types.generic_types import ValidDatatype, ValidIndex, TypeSafeElement, Index, T, K
from user_defined_types.key_types import iKey, Key
# endregion


class PriorityEntry(Generic[T, K]):
    """structure for priority queues. wraps the input data in a searchable form."""
    def __init__(self, key: K, element: T) -> None:
        self.key = key
        self.element = element

    def __iter__(self):
        yield self.key
        yield self.element

    def __lt__(self, other) -> bool:
        return self.key < other.key

    def __le__(self,other) -> bool:
        if not isinstance(other, PriorityEntry):
            return False
        return self.key <= other.key

    def __gt__(self, other) -> bool:
        return self.key > other.key

    def __ge__(self, other) -> bool:
        if not isinstance(other, PriorityEntry):
            raise DsTypeError(f"Error: Can only compare other {self.__class__.__qualname__} objects.")
        return self.key >= other.key

    def __eq__(self, other) -> bool:
        if not isinstance(other, PriorityEntry):
            raise DsTypeError(f"Error: Can only compare other {self.__class__.__qualname__} objects.")
        return self.key == other.key

    def __hash__(self) -> int:
        return hash(self.key)
