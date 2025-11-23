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
    TYPE_CHECKING
)
from abc import ABC, ABCMeta, abstractmethod

# region custom imports
from user_defined_types.generic_types import T


# endregion


"""
Linked List ADT: 
is a low level linear data structure where elements (called nodes) are connected using pointers/references rather than stored in contiguous memory like an array.

Properties:
Ordered: maintains linear sequence of elements.
Dynamic size: grows or shrinks at runtime.
Non-contiguous memory: nodes allocated individually.
Sequential access: access by index requires traversal (O(n)).
"""


# Interfaces
class LinkedListADT(ABC, Generic[T]):
    """Linked List ADT --- Canonical Operations"""

    # ----- Accessor ADT Operations -----
    @abstractmethod
    def __iter__(self) -> Generator[T, None, None]:
        """traverses all the nodes and returns the values. lazy loading via generator & yield"""
        pass
    
    @property
    @abstractmethod
    def head(self) -> Optional["iNode[T]"]:
        """returns the first node of the linked list for use as a reference."""
        pass
    
    @property
    @abstractmethod
    def tail(self) -> Optional["iNode[T]"]:
        """returns the last node of the linked list for use as a reference"""
        pass


    # ----- Mutator ADT Operations -----
    @abstractmethod
    def insert_head(self, element: T) -> "iNode[T]":
        """O(1) -- we just update the head pointer. (sometimes this is called poll)"""
        pass

    @abstractmethod
    def insert_tail(self, element: T) -> "iNode[T]":
        """O(N) -- in a singly linked list (unless you maintain a tail pointer) - sometimes this is called offer"""
        pass

    @abstractmethod
    def insert_after(self, node: "iNode[T]", element: T) -> "iNode[T]":
        """Inserts a new element value & node, after a user specified node reference."""
        pass

    @abstractmethod
    def insert_before(self, node: "iNode[T]", element: T) -> "iNode[T]":
        """Inserts a new Element(value) and node before a specified node reference."""
        pass

    @abstractmethod
    def replace(self, node: "iNode[T]", element: T) -> T:
        """replaces the element(value) at a specific node reference. - returns the old (replaced) value!"""
        pass

    @abstractmethod
    def delete(self, node: "iNode[T]") -> Optional[T]:
        """removes a specific Node via node reference O(1) - Use a doubly linked list."""
        pass


class iNode(ABC, Generic[T]):
    """Node Interface for linked list"""

    @property
    @abstractmethod
    def element(self) -> T:
        pass
    @element.setter
    @abstractmethod
    def element(self, value: T) -> None:
        pass

    @property
    @abstractmethod
    def next(self) -> Optional["iNode[T]"]:
        pass
    @next.setter
    @abstractmethod
    def next(self, value) -> None:
        pass
