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
from utils.validation_utils import DsValidation
from utils.representations import BinaryHeapRepr
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


class BinaryHeap(PriorityQueueADT[T, K], CollectionADT[T], Generic[T, K]):
    """
    Array Binary Heap 
    Sorted by Generic Priority value. (max or min) 
    this utilizes a PriorityEntry Class wrapper, that operates similar to a tuple, and stores a Key() object and the element value. 
    The Key() allows for comparisions. Any comparisons of the PriorityEntry() Object will operate on the Key()
    can choose via boolean between min or max
    The first Key() object will set the tables Keytype - every Key() must have the same type. (for consistency)
    """

    def __init__(self, datatype: type, capacity: int, min_heap: bool = False) -> None:
        self._datatype = datatype
        self._capacity = max(MIN_PQUEUE_CAPACITY, capacity)
        self._pqueue_keytype: None | type = None
        self._data = VectorArray(self._capacity, PriorityEntry)
        self._min_heap = min_heap

        # composed objects
        self._utils: PriorityQueueUtils = PriorityQueueUtils(self)
        self._validators: DsValidation = DsValidation()
        self._desc: BinaryHeapRepr = BinaryHeapRepr(self)

    @property
    def data(self) -> VectorArray[PriorityEntry]:
        return self._data

    @property
    def heap_type(self) -> bool:
        return self._min_heap

    @property
    def datatype(self) -> type:
        return self._datatype

    @property
    def keytype(self) -> Optional[type]:
        return self._pqueue_keytype

    @property
    def pqueue_size(self) -> int:
        return self._data.size

    @property
    def capacity(self) -> int:
        return self._data.capacity

    @property
    def priority(self) -> T:
        return self.find_extreme()

    # ----- Meta Collection ADT Operations -----
    def __len__(self) -> int:
        return self.pqueue_size

    def __contains__(self, value: T) -> bool:
        for i in range(self.pqueue_size):
            priority, element = self._data.array[i]
            if element == value:
                return True
        return False

    def clear(self) -> None:
        self._data.clear()
        self._data = VectorArray(self._capacity, PriorityEntry)

    def is_empty(self) -> bool:
        return self._data.is_empty()

    def __iter__(self):
        for i in range(self.pqueue_size):
            priority, element = self._data.array[i]
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
        self._utils.check_empty_pq()
        priority, element = self._data.array[0]
        return element

    # ----- Mutator ADT Operations -----
    def insert(self, element, priority) -> None:
        """
        Elements bubble up until heap-order is restored.
        """
        self._utils.check_element_already_exists(element)
        element = TypeSafeElement(element, self.datatype)
        priority = Key(priority)
        self._utils.check_key_is_same_type(priority)
        kv_pair = PriorityEntry(priority, element)
        self._data.append(kv_pair)
        self._utils.bubble_up_heap(self.pqueue_size - 1)   # starts from last element

    def extract_extreme(self) -> T:
        """
        Elements bubble down, until heap-order is restored -- O(log n)
        """
        # empty case:
        self._utils.check_empty_pq()
        # store root and last kv pairs
        root_priority, root_element = self._data.array[0]
        last = self._data.array[self.pqueue_size - 1]
        # delete the last kv pair
        last_element = self._data.delete(self.pqueue_size-1)
        if self.pqueue_size > 0:
            # swap root with last kv pair
            self._data.array[0] = last
            # restore heap order (start from root.)
            self._utils.bubble_down_heap(0)
        return root_element

    def change_priority(self, element, priority) -> None:
        """
        changes the priority of a specified element.
        remove the target element,
        reinsert new value
        recalculate heap order. bubble if necessary
        """
        self._utils.check_empty_pq()
        element = TypeSafeElement(element, self.datatype)
        priority = Key(priority)
        kv_pair = PriorityEntry(priority, element)

        found = False   # check if element found.
        for i in range(self.pqueue_size):
            stored_priority, stored_element = self._data.array[i]
            if element == stored_element:
                self._data.array[i] = kv_pair
                self._utils.bubble_up_heap(i)
                self._utils.bubble_down_heap(i)
                found = True
                return
        if not found:
            raise KeyInvalidError("Error: Element not found in Priority Queue...")


# main -- client facing code --


def main():
    heap = BinaryHeap(str, 5)
    print(f"The Heap Type is Min?: {heap.heap_type}")
    print(f"The total size of the Heap is: {len(heap)}")
    print(f"Is the heap empty? {heap.is_empty()}")
    print(heap)
    print(repr(heap))

    try:
        print(f"\nAttempting to Extract from an empty list...")
        heap.extract_extreme()
    except Exception as e:
        print(e)

    try:
        print(f"\nAttempting to Peek from an empty list...")
        heap.extract_extreme()
    except Exception as e:
        print(e)

    try:
        print(f"\nAttempting to Peek via property from an empty list...")
        heap.priority
    except Exception as e:
        print(e)

    print(f"\nInserting Items into Heap...")
    heap.insert("pay bill", 9)
    heap.insert("fix tax", 3)
    heap.insert("goto sleep", 6)
    heap.insert("floss teeth", 8)
    heap.insert("brush teeth", 2)
    heap.insert("drink water", 1)
    heap.insert("go jogging", 3)
    heap.insert("eat breakfast", 4)
    print(heap)
    print(f"\nExtracting Priority Element:")
    top = heap.extract_extreme()
    print(f"Priority Element: {top}")
    print(heap)
    print(repr(heap))

    print(f"\nPeek at Priority Element:")
    print(f"Priority Element: {heap.find_extreme()} and via property: {heap.priority}")

    print(f"\nChanging Priority of an item")
    heap.change_priority("goto sleep", 25)
    print(heap)
    print(repr(heap))

    try:
        print(f"\nAttempting to insert wrong type.")
        heap.insert(RandomClass("wowowow"), 25)
    except Exception as e:
        print(e)

    try:
        print(f"\nAttempting to insert wrong type for priority")
        heap.insert("stroing", "25")
    except Exception as e:
        print(e)

    print(f"\nThe total size of the Heap is: {len(heap)}")
    print(f"Is the heap empty? {heap.is_empty()}")
    print(repr(heap))

    # print(f"\nIterating through heap...")
    # for i in heap:
    #     print(i)

    print(f"Does the Heap contain this item? {'go jogging' in heap}")

    print(f"\nClearing Heap...")
    heap.clear()
    print(f"\nThe total size of the Heap is: {len(heap)}")
    print(f"Is the heap empty? {heap.is_empty()}")
    print(repr(heap))


if __name__ == "__main__":
    main()
