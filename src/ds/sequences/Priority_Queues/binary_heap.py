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
    Iterable
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
from utils.representations import BinaryHeapRepr
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
from ds.sequences.Priority_Queues.priority_queue_utils import PriorityQueueUtils

# endregion


class BinaryHeap(PriorityQueueADT[T], CollectionADT[T], Generic[T]):
    """
    Array Binary Heap 
    Sorted by Generic Priority value. (max or min)
    """

    def __init__(self, datatype: type, capacity: int, min_heap: bool = False) -> None:
        self._datatype = datatype
        self._capacity = max(4, capacity)
        self._key = lambda x: x[1]  # for tuples - compares priority element
        self._heap = VectorArray(self._capacity, tuple)
        self._min_heap = min_heap

        # composed objects
        self._utils = PriorityQueueUtils(self)
        self._validators = DsValidation()
        self._desc = BinaryHeapRepr(self)

    @property
    def heap_type(self):
        """boolean for min or max heap - info for __str__"""
        if self._min_heap:
            return f"Min Heap"
        else:
            return f"Max Heap"
    @property
    def min_heap(self):
        return self._min_heap
    @property
    def datatype(self):
        return self._datatype

    @property
    def key(self):
        return self._key

    @property
    def size(self):
        return self._heap.size

    @property
    def capacity(self):
        return self._heap.capacity

    @property
    def priority(self):
        return self.find_extreme()

    # ----- Meta Collection ADT Operations -----
    def __len__(self) -> int:
        return self.size

    def __contains__(self, value: T) -> bool:
        for i in range(self.size):
            element, priority = self._heap.array[i]
            if element == value:
                return True
        return False

    def clear(self) -> None:
        self._heap.clear()
        self._heap = VectorArray(self._capacity, tuple)

    def is_empty(self) -> bool:
        return self._heap.is_empty()

    def __iter__(self):
        for i in range(self.size):
            element, priority = self._heap.array[i]
            yield element

    # ------------ Utilities ------------
    def __str__(self) -> str:
        return self._desc.str_heap()

    def __repr__(self) -> str:
        return self._desc.repr_heap()

    # ----- Canonical ADT Operations -----
    # ----- Accessor ADT Operations -----

    def find_extreme(self) -> T:
        """returns but doesnt remove the priority element"""
        element, priority = self._heap.array[0]
        return element
        
    # ----- Mutator ADT Operations -----
    def insert(self, element: T, priority: int) -> None:
        """
        Elements bubble up until heap-order is restored.
        """
        self._validators.enforce_type(element, self._datatype)
        self._utils.check_element_already_exists(element)
        self._heap.append((element, priority))
        self._utils.bubble_up_heap(self.size - 1)   # starts from last element

    def extract_extreme(self) -> T:
        """
        Elements bubble down, until heap-order is restored
        O(log n)
        """
        # empty case:
        self._utils.check_empty_pq()
        # store root and last kv pairs
        root_element, root_priority = root = self._heap.array[0]
        last = self._heap.array[self.size - 1]
        # delete the last kv pair
        last_element = self._heap.delete(self.size-1)
        if self.size > 0:
            # swap root with last kv pair
            self._heap.array[0] = last
            # restore heap order
            self._utils.bubble_down_heap(0)
        return root_element
        
    def change_priority(self, element: T, priority: int) -> None:
        """
        changes the priority of a specified element.
        remove the target element,
        reinsert new value
        recalculate heap order. bubble if necessary
        """
        self._utils.check_empty_pq()
        self._validators.enforce_type(priority, int)

        found = False

        for i in range(self.size):
            stored_element, stored_priority = kv_pair = self._heap.array[i]
            if element == stored_element:
                self._heap.array[i] = (element, priority)
                self._utils.bubble_up_heap(i)
                self._utils.bubble_down_heap(i)
                found = True
                return
        
        if not found:
            raise KeyInvalidError("Error: Element not found in Priority Queue...")

            

        
# main -- client facing code -- 


def main():
    heap = BinaryHeap(str, 5)
    print(heap.heap_type)
    print(heap)
    print(repr(heap))
    heap.insert("pay bill", 9)
    heap.insert("fix tax", 3)
    heap.insert("goto sleep", 6)
    heap.insert("floss teeth", 8)
    heap.insert("brush teeth", 2)
    heap.insert("drink water", 1)
    heap.insert("go jogging", 3)
    heap.insert("eat breakfast", 4)
    print(heap)
    top = heap.extract_extreme()
    print(top)
    print(heap)
    print(heap.find_extreme())
    heap.change_priority("goto sleep", 25)
    print(heap)




if __name__ == "__main__":
    main()