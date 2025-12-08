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
    TYPE_CHECKING,
)
from abc import ABC, ABCMeta, abstractmethod
from array import array
import numpy
import ctypes
import random
import math
from collections.abc import Sequence

# endregion


# region custom imports
from user_defined_types.generic_types import T, K, iKey
from utils.exceptions import *
from utils.helpers import Ansi

if TYPE_CHECKING:
    from adts.collection_adt import CollectionADT

from ds.primitives.arrays.dynamic_array import VectorArray
from ds.sequences.Stacks.array_stack import ArrayStack
from ds.sequences.Deques.circular_array_deque import CircularArrayDeque

from user_defined_types.graph_types import weight, VertexColor

# endregion


class GraphUtils:
    """A set of Utilities for Graph Data structures"""
    def __init__(self, graph_obj) -> None:
        self.obj = graph_obj

    def view_adjacency_map(self):
        """
        prints all the vertices and their neighbours in an adjacency map form.
        {vertex: {neighbour: edge_weight, ...n-1}
        """

        curly_front, curly_back = "{", "}"
        begin = curly_front + "\n"
        end = "\n" + curly_back

        def _generate_verts(adjacency_map_items: dict):
            """Loops through all the vertices in the graph - and yields a string of neighbours and edges for easy visualization"""
            for vertex, neighbours_map in adjacency_map_items:
                # vertex identifier
                label = vertex.name
                if label is None:
                    vertex = f"{vertex}"
                else:
                    vertex = f"{label}={vertex}"

                edges_stack = ArrayStack(str)
                # loop through neighbours and get vert and edge info
                for neighbour, weight in neighbours_map.items():
                    # neighbour vertex identifier
                    label = neighbour.name
                    if label is None:
                        neighbour = f"{neighbour}"
                    else:
                        neighbour = f"{label}={neighbour}"

                    # weight identifier
                    weight = weight.element # unpacking Edge object to use weight value.
                    if weight is not None:
                        edges_stack.push(f"'{neighbour}': {weight}")
                    else:
                        edges_stack.push(f"'{neighbour}'")
                # create final string
                vertex_string = f"  '{vertex}': {curly_front}{', '.join(edges_stack)}{curly_back},"
                yield vertex_string

        out_vertices = f"\n".join(_generate_verts(self.obj._out_adj_map.items()))
        out_title = f"Graph Adjacency Map: Outgoing Edges\n" if self.obj.is_directed else f"Graph Adjacency Map: \n"

        in_vertices = f"\n".join(_generate_verts(self.obj._inc_adj_map.items()))
        in_title = f"Graph Adjacency Map: Incoming Edges\n"

        out_infostring = f"{out_title}{begin}{out_vertices}{end}\n"
        in_infostring = f"{in_title}{begin}{in_vertices}{end}\n"
        combined = f"{out_infostring}{in_infostring}"

        return combined if self.obj.is_directed else out_infostring

    def convert_adj_map_to_adjacency_matrix(self):
        """Converts an adjacency map to an adjacency matrix"""
        pass

    