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


T = TypeVar("T")


class QueueADT(ABC, Generic[T]):
    """"""
    # ----- Canonical ADT Operations -----
    @abstractmethod
    def enqueue(self, value: T):
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

    # ----- Meta Collection ADT Operations -----
    @abstractmethod
    def is_empty(self) -> bool:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass

    @abstractmethod
    def __contains__(self, value: T) -> bool:
        pass

    @abstractmethod
    def __iter__(self) -> Generator[T, None, None]:
        pass
