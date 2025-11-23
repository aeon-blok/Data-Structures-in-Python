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
from adts.priority_queue_adt import (
    MinPriorityQueueADT,
    MaxPriorityQueueADT,
    PriorityQueueADT,
)
from adts.sequence_adt import SequenceADT


from ds.primitives.arrays.dynamic_array import VectorArray, VectorView
from ds.trees.Priority_Queues.priority_queue_utils import PriorityQueueUtils

# endregion


class SortedPriorityQueue(MaxPriorityQueueADT[T], CollectionADT[T], Generic[T]):
    """
    Sorted Priority Queue:
    Array Based
    Sorted by Max Priority value.
    """
    def __init__(self, datatype: type, capacity: int) -> None:
        self._datatype = datatype
        self._capacity = min(4, capacity)
        self._key = lambda x: x[1]  # for tuples - compares priority element
        self._data = VectorArray(self._capacity, tuple)

        # composed objects
        self._utils = PriorityQueueUtils(self)
        self._validators = DsValidation()
        self._desc = PQueueRepr(self)

    @property
    def datatype(self):
        return self._datatype

    @property
    def key(self):
        return self._key

    @property
    def size(self):
        return self._data.size

    @property
    def capacity(self):
        return self._data.capacity

    @property
    def priority(self):
        return self.find_max() 

    # ----- Meta Collection ADT Operations -----
    def __len__(self) -> int:
        return self.size

    def __contains__(self, value: T) -> bool:
        for i in range(self.size):
            element, priority = self._data.array[i]
            if element == value:
                return True
        return False

    def clear(self) -> None:
        self._data.clear()
        self._data = VectorArray(self._capacity, tuple)

    def is_empty(self) -> bool:
        return self._data.is_empty()

    def __iter__(self):
        for i in range(self.size):
            element, priority = self._data.array[i]
            yield element

    # ------------ Utilities ------------
    def __str__(self) -> str:
        return self._desc.str_simple_pq()

    def __repr__(self) -> str:
        return self._desc.repr_simple_pq()

    # ----- Canonical ADT Operations -----
    # ----- Accessor ADT Operations -----
    def find_max(self) -> T:
        """retrieve but dont remove the priority element of the priority queue"""
        self._utils.check_empty_pq()
        element, priority = self._data.array[0]
        return element

    # ----- Mutator ADT Operations -----
    def insert(self, element: T, priority: int) -> None:
        """insert a key value pair into the priority queue."""
        self._validators.enforce_type(element, self._datatype)
        self._validators.enforce_type(priority, int)
        self._utils.check_element_already_exists(element)
        new_element = (element, priority)
        # empty case
        if self.is_empty():
            self._data.append(new_element)
            return
        self._utils.add_kv_pair_to_max_sorted_list(element, priority)
       
    def extract_max(self) -> T:
        """retrieve and remove the priority element"""
        # always element 0? since its a sorted list
        self._utils.check_empty_pq()
        element, priority = self._data.array[0]
        self._data.delete(0)
        return element

    def increase_key(self, element: T, priority: int) -> None:
        """
        increase the priority level for a user specified element
        In a sorted array, you must: 
            1.) remove the element 
            2.) re-insert at the correct position, (or swap until it reaches its sorted spot.)
        """
        # empty case:
        self._utils.check_empty_pq()
        self._validators.enforce_type(element, self._datatype)
        self._validators.enforce_type(priority, int)

        found_match = False

        # main case:
        # traverse - remove match if found
        for i in range(self.size):
            stored_element, stored_priority = self._data.array[i]
            if element == stored_element:
                self._utils.check_new_max_priority(priority, stored_priority)
                self._data.delete(i)    # remove element
                found_match = True
                break

        if not found_match:
            raise KeyInvalidError("Error: Element not found in priority queue.")
        
        # reinsert match at the correct position
        self._utils.add_kv_pair_to_max_sorted_list(element, priority)



# Main ---- Client Facing Code -----

def main():
    pq = SortedPriorityQueue(str, 5)
    print(pq)
    print(repr(pq))
    print(pq.is_empty())

    try:
        pq.extract_max()
    except Exception as e:
        print(e)

    try:
        pq.find_max()
    except Exception as e:
        print(e)

    pq.insert("walking", 2)
    pq.insert("eating", 2)
    pq.insert("sleeping", 5)
    pq.insert("dreaming", 2)
    pq.insert("reading", 31)
    pq.insert("studying", 14)
    pq.insert("typing", 7)
    pq.insert("shopping", 9)
    pq.insert("running", 4)
    pq.insert("drinking", 3)
    pq.insert("thinking", 22)
    print(pq)
    print(pq.find_max())
    print(pq.priority)
    print(pq.extract_max())
    print(pq)
    print(pq.extract_max())
    print(pq)

    print(pq.is_empty())
    print(len(pq))
    print("walking" in pq)
    print("studying" in pq)
    print("thinking" in pq)
    print(pq)
    try:
        pq.insert(RandomClass("whyyyrtgr"), 22)
    except Exception as e:
        print(e)

    try:
        pq.increase_key("", 25)
    except Exception as e:
        print(e)

    pq.increase_key("typing", 25)
    print(pq)

    print(pq.priority)
    pq.clear()
    print(pq)
    print(repr(pq))

if __name__ == "__main__":
    main()
