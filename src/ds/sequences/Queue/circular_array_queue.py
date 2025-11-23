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
from user_defined_types.generic_types import T
from utils.validation_utils import DsValidation
from utils.representations import CircArrayQueueRepr
from utils.helpers import RandomClass

from utils.exceptions import *

from adts.collection_adt import CollectionADT
from adts.linked_list_adt import LinkedListADT, iNode
from adts.queue_adt import QueueADT

from ds.primitives.arrays.dynamic_array import VectorArray, VectorView
from ds.sequences.Queue.queue_utils import QueueUtils
# endregion


"""
Circular Buffer / Ring Buffer or Circular Array. Fixed sized queue with rotating elements.
"""


class CircularQueue(QueueADT[T], CollectionADT[T], Generic[T]):
    """
    A Queue Data Structure based on a Static Array
    some implementations overwrite old data when full, others raise an error.
    """
    def __init__(self, datatype, capacity: int = 10, overwrite: bool = False) -> None:
        self._datatype = datatype
        self._capacity = capacity
        self._front: int = 0
        self._queue_size: int = 0
        self._overwrite = overwrite

        # composed objects
        # Buffer array – a fixed-size array holding elements.
        self._buffer = VectorArray(self._capacity, self._datatype, is_static=True)
        self._utils = QueueUtils(self)
        self._validators = DsValidation()
        self._desc = CircArrayQueueRepr(self)

    @property
    def datatype(self):
        return self._datatype

    @property
    def front(self):
        if self.is_empty():
            return None
        return self._buffer.array[self._front]

    @property
    def rear(self):
        if self.is_empty():
            return None
        return self._buffer.array[(self._front + self._queue_size - 1) % self._capacity]

    @property
    def queue_size(self):
        return self._queue_size

    # ----- Meta Collection ADT Operations -----
    def __len__(self) -> int:
        return self._queue_size

    def __contains__(self, value: T) -> bool:
        return any(item ==value for item in self)

    def __iter__(self):
        """yield the elements in order starting from the front."""
        for i in range (self._queue_size):
            index = (self._front + i) % self._capacity
            yield self._buffer.array[index]

    def is_empty(self) -> bool:
        return self._queue_size == 0

    def clear(self) -> None:
        self._buffer.clear()
        self._front = 0
        self._queue_size = 0
        self._buffer = VectorArray(self._capacity, self._datatype, is_static=True)

    # ------------ Utilities ------------
    def __str__(self) -> str:
        return self._desc.str_circ_array_queue()

    def __repr__(self) -> str:
        return self._desc.repr_circ_array_queue()

    # ----- Canonical ADT Operations -----

    def enqueue(self, value: T) -> None:
        """add an element to the rear"""
        self._validators.enforce_type(value, self.datatype)  

        # derive rear
        rear = (self._front + self._queue_size) % self._capacity

        # Full Queue Case: either overwrite or raise error -- depending on boolean
        if self._queue_size >= self._capacity:
            if self._overwrite:
                self._buffer.array[rear] = value
                self._front = (self._front + 1) % self._capacity
            else:
                raise DsOverflowError("Error: Queue is Full.")
        else:
            # Main Case: insert element
            self._buffer.array[rear] = value
            self._queue_size += 1

    def dequeue(self) -> T:
        """remove an element from the front"""
        self._utils.check_empty_queue()
        # store old value
        old_value = self._buffer.array[self._front]

        # derference front index
        if self._datatype in (object, ctypes.py_object):
            self._buffer.array[self._front] = None

        # increment front
        self._front = (self._front + 1) % self._capacity
        self._queue_size -= 1

        return old_value

    def peek(self) -> T:
        """return the front element (dont remove)"""
        self._utils.check_empty_queue()
        old_value = self._buffer.array[self._front]
        return old_value


# Main --- Client Facing Code ---


def main():
    print("=== CircularQueue Test Suite ===\n")

    # --- INT Queue ---
    print("--- Testing Integer Queue ---")
    int_queue = CircularQueue(int, capacity=5)
    print(int_queue)  # should be empty

    try:
        int_queue.dequeue()
    except Exception as e:
        print(f"Dequeue on empty queue: {e}")

    for val in [1, 2, 3]:
        int_queue.enqueue(val)
        print(int_queue)

    print(f"Front: {int_queue.front}, Rear: {int_queue.rear}, Size: {int_queue.queue_size}")

    int_queue.dequeue()
    print("After one dequeue:", int_queue)

    # Overflow test
    for val in [4, 5, 6, 7]:
        try:
            int_queue.enqueue(val)
        except Exception as e:
            print(f"Overflow test: {e}")
    print(int_queue)

    # --- Overwrite behavior ---
    print("\n--- Testing Overwrite Enabled Queue ---")
    overwrite_queue = CircularQueue(int, capacity=5, overwrite=True)
    for val in [10, 20, 30, 40, 50]:
        overwrite_queue.enqueue(val)
    print("Full queue:", overwrite_queue)

    overwrite_queue.enqueue(60)  # should overwrite oldest element
    print("After overwrite:", overwrite_queue)

    # --- STRING Queue ---
    print("\n--- Testing String Queue ---")
    str_queue = CircularQueue(str, capacity=3)
    for s in ["apple", "banana", "cherry"]:
        str_queue.enqueue(s)
        print(str_queue)

    print(f"Front: {str_queue.front}, Rear: {str_queue.rear}")

    try:
        str_queue.enqueue(RandomClass("GGGGFFFF"))  # should raise type error
    except Exception as e:
        print(f"Type safety test: {e}")

    # Dequeue all elements
    while not str_queue.is_empty():
        print(f"Dequeued: {str_queue.dequeue()}", str_queue)

    # --- OBJECT Queue ---
    print("\n--- Testing Object Queue ---")

    class Dummy:
        def __init__(self, val):
            self.val = val

        def __repr__(self):
            return f"Dummy({self.val})"

    obj_queue = CircularQueue(Dummy, capacity=3)
    for i in range(3):
        obj_queue.enqueue(Dummy(i))
        print(obj_queue)

    print(f"Front: {obj_queue.front}, Rear: {obj_queue.rear}")

    # Iteration test
    print("\nIterating over object queue:")
    for item in obj_queue:
        print(item)

    # Clear queue
    obj_queue.clear()
    print("After clear:", obj_queue)

    print("\n=== CircularQueue Wrap-Around & Overwrite Stress Test ===\n")

    capacity = 5
    cq = CircularQueue(int, capacity=capacity, overwrite=True)

    print("Initial empty queue:", cq)

    # Fill the queue completely
    for i in range(capacity):
        cq.enqueue(i)
        print(f"Enqueued {i}:", cq)

    # Overwrite elements repeatedly
    for i in range(100, 110):
        cq.enqueue(i)
        print(f"Enqueued {i} (overwrite):", cq)
        print(f"Front: {cq.front}, Rear: {cq.rear}, Size: {cq.queue_size}")

    # Dequeue all elements to check final order
    print("\nDequeuing all elements:")
    while not cq.is_empty():
        val = cq.dequeue()
        print(f"Dequeued {val}:", cq)

    # Fill and partially dequeue to test wrap-around
    for i in range(200, 205):
        cq.enqueue(i)
    print("\nQueue filled again:", cq)

    for _ in range(2):
        cq.dequeue()
    print("After 2 dequeues:", cq)

    for i in range(300, 303):
        cq.enqueue(i)
        print(f"Enqueued {i}:", cq)

    # Final queue state
    print("\nFinal queue state:")
    print(cq)

if __name__ == "__main__":
    main()
