# region standard lib
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
    Tuple,
    Literal,
    Iterable,
    TYPE_CHECKING,
)

from abc import ABC, ABCMeta, abstractmethod
from array import array
import numpy
import ctypes
import secrets
import math
import random
import time
import uuid
from pprint import pprint

# endregion

# region custom imports
from user_defined_types.generic_types import T
from utils.validation_utils import DsValidation
from utils.representations import VertexRepr
from utils.exceptions import *

from adts.collection_adt import CollectionADT


from ds.primitives.arrays.dynamic_array import VectorArray, VectorView


from user_defined_types.generic_types import (
    ValidDatatype,
    TypeSafeElement,
    PositiveNumber,
)
from user_defined_types.key_types import iKey, Key
from user_defined_types.graph_types import VertexColor
# endregion


class Vertex(Generic[T]):
    """
    Vertex Node: for Graph Data Structures
    Comes with Type Enforcement, Name Alias, Unique ID 
    """
    def __init__(
            self, 
            datatype: type, 
            element: Optional[T] = None, 
            name: Optional[str] = None, 
            comparison_key: Optional[Callable] = None
            ) -> None:
        self._id = uuid.uuid4() # immutable and globally unique
        self.name = name # user-facing label/value
        self._datatype = ValidDatatype(datatype)
        self.alive = True
        if element is None: self._element = None
        else: self._element = TypeSafeElement(element, self._datatype)
        self.comparison_key = comparison_key    # custom key for comparisons logic....
        self.insert_order: Optional[int] = None # set by graph.

        # metadata for algos
        self.predecessor: Optional[Vertex] = None # different from BST predecessor - means the node that came before this one (parent...)
        self.color = VertexColor.WHITE  # DFS / BFS
        self.distance = None    # for shortest path algos
        self.component_id = None    # connected components

        # composed object
        self._desc = VertexRepr(self)



    # -------------- Vertex Properties --------------

    @property
    def element(self) -> Optional[T]:
        return self._element
    
    @element.setter
    def element(self, value: Optional[T]) -> None:
        if value is None:
            self._element = None
        else:
            self._element = TypeSafeElement(value, self._datatype)

    @property
    def datatype(self) -> type:
        return self._datatype
    
    # -------------- Hashing and comparison for hash-based collections --------------
    def __hash__(self) -> int:
        return hash(self._id) # Delegate to the key's hash


    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Vertex):
            return self._id == other._id
        return False
      
    def __lt__(self, other) -> bool:
        """Less than...."""
        # * custom key
        if self.comparison_key is not None and other.comparison_key is not None:
            if self.comparison_key != other.comparison_key:
                raise KeyInvalidError("Error: Cannot compare vertices with different comparison keys...")
            return self.comparison_key(self._element) < self.comparison_key(other._element)
        
       # * None case: 
       # value is compared by numerical size. Strings and Tuples are by Lexographic comparison
        elif self._element is None and other._element is None:
            return False  # they are equal
        elif self._element is None:
            return True   # convention: None is “smaller” than any real value
        elif other._element is None:
            return False
        
        
        # * default fallbacks
        # compares by numerical value -- strings compare lexographically. (pythons alphanumeric ordering)
        elif issubclass(self.datatype, (int, float, str)):
            return self._element < other._element
        # compare by number of elements (aka count / total elements)
        elif issubclass(self.datatype, (list, dict, set, tuple)):
            return len(self._element) < len(other._element)
        # complex number - compares an absolute version.
        elif issubclass(self.datatype, complex):
            return abs(self._element) < abs(other._element)
        
        # * invalid Case:
        else:
            raise KeyInvalidError(f"Error: Invalid Comparison for Vertex object....")

    def __le__ (self, other) -> bool:
        """Less than or Equal to..."""

        # * custom key
        if self.comparison_key is not None and other.comparison_key is not None:
            if self.comparison_key != other.comparison_key:
                raise KeyInvalidError("Error: Cannot compare vertices with different comparison keys...")
            return self.comparison_key(self._element) <= self.comparison_key(other._element)
    
        # * none case:       
        elif self._element is None and other._element is None:
            return True  # they are equal
        elif self._element is None:
            return True   # convention: None is “smaller” than any real value
        elif other._element is None:
            return False
        
        # * default fallbacks
        # compares by numerical value -- strings compare lexographically. (pythons alphanumeric ordering)
        elif issubclass(self.datatype, (int, float, str)):
            return self._element <= other._element
        # compare by number of elements (aka count / total elements)
        elif issubclass(self.datatype, (list, dict, set, tuple)):
            return len(self._element) <= len(other._element)
        # complex number - compares an absolute version.
        elif issubclass(self.datatype, complex):
            return abs(self._element) <= abs(other._element)
        
        # * Invalid Case:
        else:
            raise KeyInvalidError(f"Error: Invalid Comparison for Vertex object....")
        
    def __gt__(self, other) -> bool:
        """Greater than"""

        # * custom key
        if self.comparison_key is not None and other.comparison_key is not None:
            if self.comparison_key != other.comparison_key:
                raise KeyInvalidError("Error: Cannot compare vertices with different comparison keys...")
            return self.comparison_key(self._element) > self.comparison_key(other._element)
        
        # * none case:       
        elif self._element is None and other._element is None:
            return False  # they are equal
        elif self._element is None:
            return False   # convention: None is “smaller” than any real value
        elif other._element is None:
            return True
        
        # * default fallbacks
        # compares by numerical value -- strings compare lexographically. (pythons alphanumeric ordering)
        elif issubclass(self.datatype, (int, float, str)):
            return self._element > other._element
        # compare by number of elements (aka count / total elements)
        elif issubclass(self.datatype, (list, dict, set, tuple)):
            return len(self._element) > len(other._element)
        # complex number - compares an absolute version.
        elif issubclass(self.datatype, complex):
            return abs(self._element) > abs(other._element)
        
        # * Invalid Case:
        else:
            raise KeyInvalidError(f"Error: Invalid Comparison for Vertex object....")
    
    def __ge__(self, other) -> bool:
        """greater than or equal to comparison"""

        # * custom key
        if self.comparison_key is not None and other.comparison_key is not None:
            if self.comparison_key != other.comparison_key:
                raise KeyInvalidError("Error: Cannot compare vertices with different comparison keys...")
            return self.comparison_key(self._element) >= self.comparison_key(other._element)

        # * none case:
        elif self._element is None and other._element is None:
            return True  # they are equal
        elif self._element is None:
            return False   # convention: None is “smaller” than any real value
        elif other._element is None:
            return True
        
        # * default fallbacks
        # compares by numerical value -- strings compare lexographically. (pythons alphanumeric ordering)
        elif issubclass(self.datatype, (int, float, str)):
            return self._element >= other._element
        # compare by number of elements (aka count / total elements)
        elif issubclass(self.datatype, (list, dict, set, tuple)):
            return len(self._element) >= len(other._element)
        # complex number - compares an absolute version.
        elif issubclass(self.datatype, complex):
            return abs(self._element) >= abs(other._element)
        
        # * Invalid Case:
        else:
            raise KeyInvalidError(f"Error: Invalid Comparison for Vertex object....")

    # -------------- Utilities -----------------
    def __str__(self) -> str:
        return self._desc.str_vertex()
    
    def __repr__(self) -> str:
        return self._desc.repr_vertex()


# ------------------------ Main(): Client Facing Code --------------------------

def main():
    vert_a = Vertex(str, "the Capital!", "Berlin")
    vert_b = Vertex(str, "amsterdam")
    vert_c = Vertex(str, "Beijing", "China")

    vertices = [vert_a, vert_b, vert_c]
    sorted_verts = sorted(vertices)
    print(sorted_verts)
    print(vert_a)



if __name__ == "__main__":
    main()
