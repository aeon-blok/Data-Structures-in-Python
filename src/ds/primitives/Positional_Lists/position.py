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
    TYPE_CHECKING,
)
from abc import ABC, ABCMeta, abstractmethod

# endregion

# region custom imports
from utils.helpers import RandomClass
from user_defined_types.custom_types import T
from utils.constants import DLL_SEPERATOR
from utils.representations import PNodeRepr, PositionRepr

from adts.positional_list_adt import iNode, iPosition

if TYPE_CHECKING:
    from adts.positional_list_adt import PositionalListADT


class PNode(iNode[T]):
    def __init__(
            self, 
            element: Optional[T] = None, 
            next: Optional["iNode[T]"] = None, 
            prev: Optional["iNode[T]"] = None
            ) -> None:
        self._element = element
        self._prev = prev
        self._next = next
        # Composed Objects
        self._desc = PNodeRepr(self)

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

    # ------------ Utilities ------------
    def __repr__(self) -> str:
        return self._desc.repr_p_node()


class Position(iPosition[T]):
    def __init__(
            self, 
            node: iNode[T], 
            container: Optional["PositionalListADT[T]"] = None
            ) -> None:

        self._node = node   # underlying node reference
        self._element: Optional[T] = None
        self._container = container # represents a specific position list.
        self._access_counter: int = 0    # for MTF heuristic. (frequency based access)
        # composed objects
        self._desc = PositionRepr(self)
    @property
    def node(self):
        return self._node
    @property
    def element(self):
        """returns the underlying node element value"""
        return self._node.element
    @property
    def container(self):
        return self._container
    @container.setter
    def container(self, value) -> None:
        self._container = value

    def __repr__(self) -> str:
        return self._desc.repr_position()

    def __eq__(self, value: object) -> bool:
        """the two position objects are compared and considered equal ONLY if they point to the exact same memory address."""
        return type(value) is type(self) and value._node is self._node

    def __ne__(self, value: object) -> bool:
        """define not equivalent logic - should be the types and memory addresses dont match."""
        return not (self == value)
