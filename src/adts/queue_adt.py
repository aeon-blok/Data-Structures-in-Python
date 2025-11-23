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


# endregion

"""
Queue ADT Definition:
Formally, the queue abstract data type defines a collection that keeps objects in a sequence.
Where element access & deletion are restricted to the first element in the queue.
Element insertion is restricted to the back of the sequence.

Operations:
enqueue(Q, x): Add element x to the rear of the queue Q
dequeue(Q) -> x: remove & return the first element from the queue Q. An error will be raised if the queue Q is empty
peek(Q) -> x: return the first element (but do not remove it.)

Properties / Constraints:
FIFO principle: First element enqueued is first dequeued.
Non-commutative insertion: Order of enqueues affects order of dequeues.
Empty queue identity: A newly created queue has no elements.
Dequeue on empty queue: Raises error/exception.
"""


class QueueADT(ABC, Generic[T]):
    """
    Queue ADT: Access and Deletion of elements are restricted to the first element of the queue. (front)
    Insertion of elements is restricted to the last element of the queue (rear)
    """

    # ----- Canonical ADT Operations -----
    @abstractmethod
    def enqueue(self, value: T) -> None:
        """Adds an Element to the end of the Queue"""
        pass

    @abstractmethod
    def dequeue(self) -> T:
        """remove and return the first element of the Queue"""
        pass

    @abstractmethod
    def peek(self) -> T:
        """return (but not remove) the first element of the Queue"""
        pass
