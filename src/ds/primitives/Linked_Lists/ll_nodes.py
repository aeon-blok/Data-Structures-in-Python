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
    Iterable,
    Type,
    TYPE_CHECKING
)
from abc import ABC, ABCMeta, abstractmethod

# endregion

# region custom imports
from utils.helpers import RandomClass
from utils.custom_types import T
from utils.constants import DLL_SEPERATOR
from utils.representations import DllNodeRepr, SllNodeRepr

if TYPE_CHECKING:
    from adts.linked_list_adt import LinkedListADT

from adts.linked_list_adt import iNode


# endregion


# Concrete Classes
class Sll_Node(iNode[T]):
    """Node Element in Linked List"""

    def __init__(self, element: T, is_linked: bool = False, list_owner: Optional["LinkedListADT[T]"]=None) -> None:
        self._element = element
        # initialized as none, stores a reference to the next node in the linked list
        self._next: Optional["iNode[T]"] = None
        self._is_linked = is_linked  # checks if node is deleted or not.
        # ensures the node belongs to the correct list, preventing cross-list misuse.
        self._list_owner = list_owner

        # composed objects
        self._desc = SllNodeRepr(self)

    @property
    def element(self):
        return self._element

    @element.setter
    def element(self, value):
        self._element = value

    @property
    def next(self):
        return self._next

    @next.setter
    def next(self, value) -> None:
        self._next = value

    @property
    def is_linked(self):
        return self._is_linked

    @is_linked.setter
    def is_linked(self, value):
        self._is_linked = value

    @property
    def list_owner(self):
        return self._list_owner

    @list_owner.setter
    def list_owner(self, value):
        self._list_owner = value

    # ----- Utility Operations -----
    def __str__(self):
        return self._desc.str_ll_node()

    def __repr__(self):
        return self._desc.repr_sll_node()

class Dll_Node(iNode[T]):
    def __init__(self, element: T, is_linked: bool = False, list_owner=None) -> None:
        self._element = element
        self._prev: Optional["iNode[T]"] = None
        self._next: Optional["iNode[T]"] = None
        self._is_linked = is_linked
        self._list_owner = list_owner
        self._desc = DllNodeRepr(self)

    @property
    def prev(self):
        return self._prev

    @prev.setter
    def prev(self, value):
        self._prev = value

    @property
    def next(self):
        return self._next

    @next.setter
    def next(self, value):
        self._next = value

    @property
    def element(self):
        return self._element

    @element.setter
    def element(self, value):
        self._element = value

    @property
    def is_linked(self):
        return self._is_linked

    @is_linked.setter
    def is_linked(self, value):
        self._is_linked = value

    @property
    def list_owner(self):
        return self._list_owner

    @list_owner.setter
    def list_owner(self, value):
        self._list_owner = value

    # ------------ Utilities ------------
    def __repr__(self) -> str:
        return self._desc.repr_dll_node()

    def __str__(self) -> str:
        return self._desc.str_ll_node()
