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
from types.custom_types import T
from utils.validation_utils import DsValidation
from utils.representations import PQueueRepr
from utils.helpers import RandomClass
from utils.exceptions import *

from adts.collection_adt import CollectionADT
from adts.priority_queue_adt import MinPriorityQueueADT, MaxPriorityQueueADT, PriorityQueueADT
from adts.sequence_adt import SequenceADT


from ds.primitives.arrays.dynamic_array import VectorArray, VectorView
from ds.trees.Priority_Queues.priority_queue_utils import PriorityQueueUtils

# endregion


"""
Unsorted Priority Queue:
The items are stored unsorted. -- Insertion is fast (O(1) for array append or list add), because we don’t care about order.
Finding/removing the extreme (max or min, depending on the priority) is slow: O(n)  -- because we must scan all items to find the highest-priority one.
"""


class UnsortedMinPriorityQueue(MinPriorityQueueADT[T], CollectionADT[T], Generic[T]):
    """
    Unsorted Array Based Min Priority Queue:
    Priority Value (int): Min = top priority
    Items stored as tuples
    """
    def __init__(self, datatype: type, capacity: int = 10) -> None:
        self._datatype = datatype
        self._capacity = max(4, capacity)
        self._data = VectorArray(self._capacity, tuple)
        self._key = lambda x: x[1]  # for tuples - compares priority element.
        # composed objects
        self._utils = PriorityQueueUtils(self)
        self._validators = DsValidation()
        self._desc = PQueueRepr(self)

    @property
    def datatype(self):
        return self._datatype

    @property
    def capacity(self):
        return self._data.capacity

    @property
    def size(self):
        return self._data.size

    @property
    def priority(self):
        return self.find_min()

    # ----- Meta Collection ADT Operations -----
    def __contains__(self, value: T) -> bool:
        for i in range(self.size):
            element = self._data.array[i]
            if element == value:
                return True
        return False

    def __len__(self) -> int:
        return self.size

    def clear(self) -> None:
        self._data.clear()
        self._data = VectorArray(self._capacity, tuple)

    def __iter__(self):
        for i in range(self.size):
            kv_pair = self._data.array[i]
            element, priority = kv_pair
            yield element

    def is_empty(self) -> bool:
        return self.size == 0

    # ------------ Utilities ------------

    def __str__(self) -> str:
        return self._desc.str_simple_pq()

    def __repr__(self) -> str:
        return self._desc.repr_simple_pq()

    # ----- Canonical ADT Operations -----
    # ----- Accessor ADT Operations -----

    def find_min(self) -> T:
        """retrives the current priority element (value). (doesnt remove.) -- O(n) linear time."""
        # empty pq Case:
        self._utils.check_empty_pq()

        # Main Case: if a candidate element's priority is less than every other item in the array - it becomes the priority element.
        priority_index = self._utils.linear_scan_min(self._data)
        kv_pair = self._data.array[priority_index]
        element, priority = kv_pair
        return element

    # ----- Mutator ADT Operations -----

    def insert(self, element: T, priority: int) -> None:
        """inserts a kv pair into the priority queue"""
        self._utils.check_element_already_exists(element)
        self._validators.enforce_type(element, self._datatype)
        self._validators.enforce_type(priority, int)

        kv_pair = (element, priority)   # pack storage tuple
        self._data.append(kv_pair)  # append to array
        # should update size automatically.

    def extract_min(self) -> Optional[T]:
        """removes and returns the priority element from the priority queue"""
        self._utils.check_empty_pq()
        priority_index = self._utils.linear_scan_min(self._data)
        kv_pair = self._data.array[priority_index]
        element, priority = kv_pair
        self._data.delete(priority_index)   # handles size decrement auto
        return element

    def decrease_key(self, element: T, priority: int) -> None:
        """
        intended to lower the priority of an element.
        Many algorithms (Dijkstra, Prim, A*) assume that once an element is in the PQ, its priority can only improve, i.e., get smaller.
        """
        self._utils.check_empty_pq()
        self._validators.enforce_type(element, self._datatype)
        self._validators.enforce_type(priority, int)

        # find element.
        for i in range(self.size):
            # unpack tuple.
            stored_element, stored_priority = self._data.array[i]
            if element == stored_element:
                # ensure new priority is lower.
                self._utils.check_new_min_priority(priority, stored_priority)
                # replace stored priority with new value
                self._data.array[i] = (element, priority)
                return
            
        raise KeyInvalidError("Error: Element not found in priority queue.")


# Main ---- Client Facing Code -----


def main():
    pq =UnsortedMinPriorityQueue(str, capacity=4)
    print(pq)
    print(repr(pq))

    try:
        pq.find_min()
    except Exception as e:
        print(e)

    pq.insert("eat breakfast", 2)
    pq.insert("clean room", 7)
    pq.insert("watch movie", 1)
    pq.insert("study data structures", 0)
    pq.insert("figure out tax", 3)
    pq.insert("write movie", 1)
    pq.insert("eat lunch", 2)
    pq.insert("talk to friends", 2)
    pq.insert("read book", 5)

    print(pq)
    print(repr(pq))

    print(pq.find_min())
    print(pq.extract_min())
    print(pq)

    pq.decrease_key("clean room", 1)
    print(pq)

    try:
        pq.decrease_key(" ", 1)    
    except Exception as e:
        print(e)

    print(pq)

    try:
        pq.decrease_key("clean room", 5)
    except Exception as e:
        print(e)

    try:
        pq.insert(RandomClass("YOOOYOYOYOYOYOY"), 0)
    except Exception as e:
        print(e)

    print(f"Total Number of Items in Priority Queue: {len(pq)}")
    print(f"{'dildo' in pq}")
    print(f"{pq.is_empty()}")

    for i in pq:
        print(i)

    print(pq)
    print(pq.priority)
    pq.clear()
    print(pq)

if __name__ == "__main__":
    main()
