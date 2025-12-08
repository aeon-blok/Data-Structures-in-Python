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
    Tuple,
    Literal,
    Iterable,
    TYPE_CHECKING,
    NewType,
    Union,
)

from abc import ABC, ABCMeta, abstractmethod
from array import array
import numpy
import ctypes
import secrets
import math
import random
import time
from pprint import pprint

# endregion

# region custom imports
from user_defined_types.generic_types import T
from user_defined_types.key_types import iKey
from user_defined_types.graph_types import weight
if TYPE_CHECKING:
    from adts.collection_adt import CollectionADT

from ds.primitives.arrays.dynamic_array import VectorArray
from ds.maps.Sets.hash_set import HashSet
from ds.graphs.vertex import Vertex
from ds.graphs.edge import Edge

# endregion


"""
Graph ADT:
A graph is an ordered pair:  G = (V, E)
Where V is a non empty finite vertex set
Where E ⊆ V × V (if a directed graph) Or E ⊆ {{u, v}} (if an undirected graph)

Properties:
E <= V² (vertex edges can point to themselves)

"""


class GraphADT(Generic[T]):
    """Canonical Operations for Graph ADT"""

    # ----- Canonical ADT Operations -----

    @abstractmethod
    def has_vertex(self, vertex: Vertex) -> bool:
        """Checks whether specified vertex exists in the graph and returns a boolean"""
        ...

    @abstractmethod
    def has_edge(self, u: Vertex, v: Vertex) -> bool:
        """Checks whether an edge exists between the specified vertices and returns a boolean"""
        ...

    # ----- Accessors -----
    @property
    @abstractmethod
    def vertex_count(self) -> int:
        """returns the total number of vertices in the graph."""
        ...

    @property
    @abstractmethod
    def edge_count(self) -> int:
        """returns the total number of edges in the graph. this includes BOTH outgoing and incoming edges summed together."""
        ...

    @abstractmethod
    def neighbours(self, vertex: Vertex) -> VectorArray[Vertex]:
        """yields the immediate neighbours verts of the specified vertex. (1 edge away.)"""
        ...

    @abstractmethod
    def degree(self, vertex: Vertex) -> int:
        """returns the degree (num of edges) of the specified vertex. (for directed graphs it will be the out-degree)"""
        ...

    @abstractmethod
    def incident_edges(self, vertex: Vertex) -> VectorArray[Edge]:
        """returns all edges incident to the specified vertex. report outgoing edges by default, with the option report ingoing edges."""
        ...

    @abstractmethod
    def get_edge(self, u: Vertex, v: Vertex) -> Edge:
        """finds and returns the edge from vertex u, v if it exists otherwise returns none."""
        ...

    # ----- Mutators -----
    @abstractmethod
    def add_vertex(self, element: T, label: Optional[str]=None, vertex_comparison_key: Optional[Callable]=None) -> Vertex:
        """Adds a new Vertex to the graph. the input value is the element of the vertex object."""
        ...

    @abstractmethod
    def remove_vertex(self, vertex: Vertex) -> None:
        """Removes a Vertex and ALL incident edges from the graph."""
        ...

    @abstractmethod
    def add_edge(self, u: Vertex, v: Vertex, element: Optional[weight] = None) -> Edge:
        """Adds an edge from vertex U to vertex V. (either directed or undirected.) With the option to set a weight. (for weighted graphs)"""
        ...

    @abstractmethod
    def remove_edge(self, u: Vertex, v: Vertex) -> None:
        """removes the edge between vertex U and vertex V."""
        ...
