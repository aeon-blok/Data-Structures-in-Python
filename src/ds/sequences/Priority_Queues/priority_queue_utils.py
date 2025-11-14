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
)
from abc import ABC, ABCMeta, abstractmethod
from array import array
import numpy
import ctypes
import random
from collections.abc import Sequence

# endregion


# region custom imports
from utils.custom_types import T
from utils.validation_utils import DsValidation
from utils.exceptions import *

if TYPE_CHECKING:
    from adts.collection_adt import CollectionADT
    from adts.priority_queue_adt import PriorityQueueADT

# endregion

class PriorityQueueUtils:
    """Util Methods for Priority Queues"""
    def __init__(self, priority_queue_obj) -> None:
        self.obj = priority_queue_obj

    def check_empty_pq(self):
        if self.obj.is_empty():
            raise DsUnderflowError("Error: Priority Queue is Empty.")

    def check_element_already_exists(self, element):
        if element in self.obj:
            raise DsDuplicationError("Error: Element already exists. Use 'Decrease Key()' to modify priority level.")

    def check_new_min_priority(self, new_priority: int, stored_priority: int):
        if new_priority > stored_priority:
            raise PriorityInvalidError("Error: Priority input must be lower than currently stored priority value.")

    def check_new_max_priority(self, new_priority: int, stored_priority: int):
        if new_priority < stored_priority:
            raise PriorityInvalidError("Error: Priority input must be higher than currently stored priority value.")

    def linear_scan_min(self, input_array) -> int:
        """Linear Scan: compare to all other elements in the array."""
        candidate = input_array.array[0]
        priority_index = 0
        for i in range(self.obj.size):
            kv_pair = self.obj._data.array[i]
            if self.obj._key(kv_pair) < self.obj._key(candidate):
                candidate = kv_pair
                priority_index = i
        return priority_index

    def linear_scan_max(self, input_array) -> int:
        """Linear Scan: compare to all other elements in the array."""
        candidate = input_array.array[0]
        priority_index = 0
        for i in range(self.obj.size):
            kv_pair = self.obj._data.array[i]
            if self.obj._key(kv_pair) > self.obj._key(candidate):
                candidate = kv_pair
                priority_index = i
        return priority_index

    def add_kv_pair_to_max_sorted_list(self, element, priority):
        """Automatically adds the item to the correct spot in the list."""
        kv_pair = (element, priority)
        found = False
        # traverse through items.
        for i in range(self.obj.size):
            current_element, current_priority = self.obj._data.array[i]
            if priority > current_priority:
                self.obj._data.insert(i, kv_pair)
                return
        # lowest priority case: -- if priority is the lowest - add to the end of the array
        self.obj._data.append(kv_pair)
