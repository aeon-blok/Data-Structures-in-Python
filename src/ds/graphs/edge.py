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
from utils.representations import EdgeRepr
from utils.exceptions import *

from adts.collection_adt import CollectionADT



from ds.primitives.arrays.dynamic_array import VectorArray, VectorView


from user_defined_types.generic_types import (
    ValidDatatype,
    TypeSafeElement,
    PositiveNumber,
)
from user_defined_types.key_types import iKey, Key
from user_defined_types.graph_types import VertexColor, ValidVertex

from ds.graphs.vertex import Vertex

# endregion



class Edge(Generic[T]):
    """Edge Object for graphs. it stores references to the vertices as endpoints"""

    def __init__(self, dataype: type, u: Vertex, v: Vertex, element: Optional[T], comparison_key: Optional[Callable] = None) -> None:
        self._id = uuid.uuid4() # immutable and globally unique
        self._datatype = ValidDatatype(dataype)
        self._origin = ValidVertex(u, Vertex)
        self._destination = ValidVertex(v, Vertex)
        if element is None: self._element = None
        else: self._element = TypeSafeElement(element, self._datatype)
        self.insert_order: Optional[int] = None  # set by graph. (couples edge to this specific graph)
        self.comparison_key = comparison_key    # custom key for comparisons logic....
        self._desc = EdgeRepr(self)
             
    @property
    def datatype(self) -> type:
        return self._datatype

    @property
    def element(self) -> Optional[T]:
        return self._element
   
    @element.setter
    def element(self, value: T) -> None:
        if value is None:
            self._element = None
        else:
            self._element = TypeSafeElement(value, self._datatype)
    
    @property
    def origin(self) -> Vertex:
        return self._origin
    
    @property
    def destination(self) -> Vertex:
        return self._destination
        

    def __str__(self) -> str:
        return self._desc.str_edge()
  
    def __repr__(self) -> str:
        return self._desc.repr_edge()

    # ----- Accessors -----
    def endpoints(self):
        """Returns a tuple of the vertices endpoints (u, v)"""
        return (self._origin, self._destination)
    
    def opposite(self, v: Vertex):
        """returns the vertex that is opposite the input vertex (v) on this edge."""
        return self._destination if v is self._origin else self._origin

    # -------------- Hashing and comparison for hash-based collections --------------
    def __hash__(self) -> int:
        """this allows an edge to be used as a key in a map or a set."""
        return hash(self._id)
    
    def __eq__(self, other) -> bool:
        return hash(self._id) == hash(other._id)
    
    def __lt__(self, other) -> bool:
        """less than: uses the element value of the edge. Since edge can be none. there are additional checks required."""
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
        """Less than or Equal to.... """
        # * custom key
        if self.comparison_key is not None and other.comparison_key is not None:
            if self.comparison_key != other.comparison_key:
                raise KeyInvalidError("Error: Cannot compare vertices with different comparison keys...")
            return self.comparison_key(self._element) <= self.comparison_key(other._element)
        
        # * None case: 
        # value is compared by numerical size. Strings and Tuples are by Lexographic comparison
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
        
        # * invalid Case:
        else:
            raise KeyInvalidError(f"Error: Invalid Comparison for Vertex object....")

    def __gt__(self, other) -> bool:
        """Greater than"""
        # * custom key
        if self.comparison_key is not None and other.comparison_key is not None:
            if self.comparison_key != other.comparison_key:
                raise KeyInvalidError("Error: Cannot compare vertices with different comparison keys...")
            return self.comparison_key(self._element) > self.comparison_key(other._element)
        
        # * None case: 
        # value is compared by numerical size. Strings and Tuples are by Lexographic comparison
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
        
        # * invalid Case:
        else:
            raise KeyInvalidError(f"Error: Invalid Comparison for Vertex object....")
    
    def __ge__(self, other) -> bool:
        """Greater than or Equal To.."""
        # * custom key
        if self.comparison_key is not None and other.comparison_key is not None:
            if self.comparison_key != other.comparison_key:
                raise KeyInvalidError("Error: Cannot compare vertices with different comparison keys...")
            return self.comparison_key(self._element) >= self.comparison_key(other._element)
        
        # * None case: 
        # value is compared by numerical size. Strings and Tuples are by Lexographic comparison
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
        
        # * invalid Case:
        else:
            raise KeyInvalidError(f"Error: Invalid Comparison for Vertex object....")
    





# ------------------------ Main(): Client Facing Code --------------------------

def main():
    vert_a = Vertex(str, "alien")
    vert_b = Vertex(str, "rambo")
    edg_a = Edge(float, vert_a, vert_b, 2.5)

    print(edg_a)
    print(repr(edg_a))






if __name__ == "__main__":
    main()




























"""
Edge:
reference to the vertex objects. at either endpoint of the edge.
reference to the element of edge (default to weight?)
"""

