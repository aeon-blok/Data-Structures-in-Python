from typing import (
    Generic,
    TypeVar,
    List,
    Dict,
    Optional,
    Callable,
    Any,
    cast,
    Generator,
    Iterator,
    Iterable,
)
from abc import ABC, ABCMeta, abstractmethod


"""
**Linked List** is a linear data structure where elements (called nodes) are connected using pointers/references rather than stored in contiguous memory like an array.

Properties:
Ordered: maintains linear sequence of elements.
Dynamic size: grows or shrinks at runtime.
Non-contiguous memory: nodes allocated individually.
Sequential access: access by index requires traversal (O(n)).
"""

T = TypeVar("T")


# Interfaces
class iLinkedList(ABC, Generic[T]):
    """"""
    
    @abstractmethod
    def insert_head(self, node_data):
        """O(1) -- we just update the head pointer."""
        pass

    @abstractmethod
    def insert_tail(self, node_data):
        """O(N) -- in a singly linked list (unless you maintain a tail pointer)"""
        pass

    @abstractmethod
    def insert_at(self, index: int, node_data):
        """O(N) -- Inserting elsewhere, because you have to traverse (for loop)."""
        pass

    @abstractmethod
    def delete_at(self, index: int) -> Optional[T]:
        """O(N) -- because you have to traverse the linked list (via .next)"""
        pass

    @abstractmethod
    def delete_head(self) -> Optional[T]:
        """O(1) -- we just remove the head."""
        pass

    @abstractmethod
    def search_by_index(self, index: int) -> Optional[T]:
        pass

    @abstractmethod
    def search_for_index(self, node_data) -> int | None:
        pass

    @abstractmethod
    def contains(self, node_data) -> bool:
        pass

    @abstractmethod
    def traverse(self, function: Callable) -> Generator["Node" | T, None, None]:
        pass

    @abstractmethod
    def is_empty(self) -> bool:
        pass

    @abstractmethod
    def length(self) -> int:
        pass
