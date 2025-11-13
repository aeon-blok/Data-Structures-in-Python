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
from utils.custom_types import T
from utils.validation_utils import DsValidation
from utils.representations import llQueueRepr
from utils.helpers import RandomClass

from utils.exceptions import *

from adts.collection_adt import CollectionADT
from adts.linked_list_adt import LinkedListADT, iNode
from adts.queue_adt import QueueADT

from ds.primitives.Linked_Lists.ll_nodes import Sll_Node
from ds.sequences.Queue.queue_utils import QueueUtils
from ds.primitives.Linked_Lists.sll import LinkedList

# endregion


class LlQueue(QueueADT[T], CollectionADT[T], Generic[T]):
    """Linked List Queue Implementation. O(1) - insert and retrieval & deletion"""
    def __init__(self, datatype: type) -> None:
        self._datatype = datatype

        # composed objects
        self._ll = LinkedList(self._datatype)
        self._utils = QueueUtils(self)
        self._validator = DsValidation()
        self._desc = llQueueRepr(self)

    @property
    def front(self):
        self._utils.check_empty_queue()
        result = self._ll.head.element
        return result
    @property
    def rear(self):
        self._utils.check_empty_queue()
        result = self._ll.tail.element
        return result
    @property
    def datatype(self):
        return self._datatype
    @property
    def size(self):
        return self._ll.total_nodes

    # ----- Meta Collection ADT Operations -----
    def __len__(self) -> int:
        return self._ll.total_nodes

    def __contains__(self, value: T) -> bool:
        return self._ll.__contains__(value)

    def is_empty(self) -> bool:
        return self._ll.is_empty()

    def __iter__(self):
        return self._ll.__iter__()

    def clear(self) -> None:
        return self._ll.clear()

    # ------------ Utilities ------------
    def __str__(self) -> str:
        return self._desc.str_ll_queue()

    def __repr__(self) -> str:
        return self._desc.repr_ll_queue()

    # ----- Canonical ADT Operations -----
    def enqueue(self, value: T):
        """inserts an element at the back of the queue."""
        self._validator.enforce_type(value, self._datatype)
        self._ll.insert_tail(value)

    def dequeue(self) -> T:
        """removes and returns the front element of the queue"""
        self._utils.check_empty_queue()
        old_value = self._ll.head.element
        self._ll.delete_head()
        return old_value

    def peek(self) -> T:
        """returns the front of the queue value (but doesnt remove it)"""
        self._utils.check_empty_queue()
        result = self._ll.head.element
        return result 


# Main --- Client Facing Code ---
def main():
    print("=== LlQueue Test Suite with __str__ ===\n")

    queue = LlQueue(int)

    # --- Empty queue ---
    print("Initial empty queue:")
    print(queue, "\n")

    # --- Enqueue operations ---
    print("Enqueue Operations:")
    for val in [10, 20, 30, 40]:
        queue.enqueue(val)
        print(queue)

    # --- Peek operation ---
    print("\nPeek front:")
    try:
        print(queue.peek())
    except IndexError as e:
        print(e)

    # --- Dequeue operations ---
    print("\nDequeue Operations:")
    while not queue.is_empty():
        queue.dequeue()
        print(queue)

    # --- Type Safety Test ---
    print("\nType Safety Test:")
    try:
        queue.enqueue(RandomClass("woooowllololo"))
    except Exception as e:
        print("Caught type error:", e)

    queue.enqueue(50)
    queue.enqueue(60)
    print(queue)

    # --- Error handling: Dequeue/peek/front/rear on empty queue ---
    print("\nError Handling Test:")
    queue.clear()
    print(queue)  # empty queue
    for method in [queue.dequeue, queue.peek, lambda: queue.front, lambda: queue.rear]:
        try:
            method()
        except Exception as e:
            print("Caught error:", e)
if __name__ == "__main__":
    main()
