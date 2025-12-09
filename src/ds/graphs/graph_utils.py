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
from ds.sequences.Deques.linked_list_deque import DllDeque
from ds.sequences.Deques.circular_array_deque import CircularArrayDeque
from ds.maps.Sets.hash_set import HashSet
from ds.graphs.vertex import Vertex


from user_defined_types.graph_types import weight, VertexColor, ValidVertex
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

    def dfs_combined_iterative_traversal(self, source_vertex: Vertex):
        """
        DFS - returns preorder and postorder traversal
        Precondition: a start vertex is supplied
        Postcondition: visits reachable vertices only
        """

        # Validate Inputs
        source_vertex = ValidVertex(source_vertex, Vertex)

        # initialize stack
        stack = ArrayStack(Vertex)
        reverse_stack = ArrayStack(Vertex)
        # set - checks whether a vertex has already been visited. -- O(1) membership checks.
        visited = HashSet(Vertex)
        preorder = VectorArray(100, Vertex)
        postorder = VectorArray(100, Vertex)

        # * mark initial vertex as visited
        visited.add(source_vertex)
        stack.push(source_vertex)

        # * iterate over stack - mark each node as visited and move to its unvisited neighbours.
        while stack:
            vertex = stack.pop()
            preorder.append(vertex)
            reverse_stack.push(vertex)
            neighbours = self.obj.neighbours(vertex)
            # reversed - this is preordering.
            for i in reversed(neighbours):
                if i not in visited:
                    visited.add(i)
                    stack.push(i)

        # 2 stacks - for postorder.
        while reverse_stack:
            postorder.append(reverse_stack.pop())

        return preorder, postorder 

    def dfs_preorder_iterative_traversal(self, source_vertex: Vertex, reverse_preorder=False):
        """depth first search - uses stack and iterative traversal. Preorder implementation (first to last visited...)"""

        # Validate Inputs
        source_vertex = ValidVertex(source_vertex, Vertex)

        # initialize stack
        stack = ArrayStack(Vertex)
        # set - checks whether a vertex has already been visited. -- O(1) membership checks.
        visited = HashSet(Vertex)
        # preorder array - nodes are added in order of discovery, from first to last.
        preorder = VectorArray(100, Vertex)

        # * mark initial vertex as visited
        visited.add(source_vertex)
        stack.push(source_vertex)

        # * iterate over stack - mark each node as visited and move to its unvisited neighbours.
        while stack:
            vertex = stack.pop()
            preorder.append(vertex)
            neighbours = self.obj.neighbours(vertex)
            # reversed - this is preordering.
            for i in reversed(neighbours):
                if i not in visited:
                    visited.add(i)
                    stack.push(i)

        # * if reverse preorder - reverse the preorder array and return.
        # (NOTE: reverse preorder is NOT the same as postorder)
        if reverse_preorder:
            rev_preorder = VectorArray(preorder.size, Vertex)
            for i in reversed(preorder):
                rev_preorder.append(i)
            return rev_preorder

        # return the preorder array of vertices.
        return preorder 

    def dfs_postorder_iterative_traversal(self, source_vertex: Vertex, reverse_postorder=False):
        """
        postorder implementation of DFS (last to first...) 
        Utilize the two stack method to get postorder for nodes.
        """

        # Validate Inputs
        source_vertex = ValidVertex(source_vertex, Vertex)

        # initialize stack
        stack = ArrayStack(Vertex)
        reverse_stack = ArrayStack(Vertex)
        # set - checks whether a vertex has already been visited. -- O1 membership checks.
        visited = HashSet(Vertex)

        # * mark initial vertex as visited
        visited.add(source_vertex)
        stack.push(source_vertex)

        # * iterate over stack - mark each node as visited and move to its unvisited neighbours.
        while stack:
            vertex = stack.pop()
            reverse_stack.push(vertex)
            neighbours = self.obj.neighbours(vertex)
            # postorder does not use reversed...
            for i in neighbours:
                if i not in visited:
                    visited.add(i)
                    stack.push(i)

        # Postorder array - utilizes our second stack which reverses the ordering due to its LIFO nature
        postorder = VectorArray(reverse_stack.size, Vertex)
        while reverse_stack:
            postorder.append(reverse_stack.pop())

        # reverse the array for reverse Postorder - NOT the same as preorder.
        if reverse_postorder:
            rev_postorder = VectorArray(postorder.size, Vertex)
            for i in reversed(postorder):
                rev_postorder.append(i)
            return rev_postorder

        # return the postorder array of vertices.
        return postorder

    def bfs_iterative_traversal(self, source_vertex: Vertex):
        """BFS implementation using a deque and iterative traversal..."""
        # init containers
        source_vertex = ValidVertex(source_vertex, Vertex)
        bfs_queue = CircularArrayDeque(Vertex, 100)
        visited = HashSet(Vertex)
        levelorder = VectorArray(100, Vertex)

        # add source vertex to the deque...
        bfs_queue.add_front(source_vertex)
        # invariant: A vertex must be marked visited at the moment it is first discovered (enqueued).
        visited.add(source_vertex)

        # traverse entire graph. starting from source node, add each node to the visited set to prevent infinite recursion
        # append the nodes to the level order array to get an ordered list, (shortest distance from source vertex to furthest distance...)
        while bfs_queue:
            vertex = bfs_queue.remove_front()
            levelorder.append(vertex)
            for i in self.obj.neighbours(vertex):
                if i not in visited:
                    bfs_queue.add_rear(i)
                    visited.add(i)
        return levelorder

    def dfs_forest(self):
        """
        A DFS forest is the union of DFS trees, one per connected component.
        dfs over all connected components in the graph.
        iterative variety with stack....
        a MD array or matrix is the returned result containing arrays of all connected graphs and their order...
        This is a Connected Components Algorithm in practice.
        """
        visited = HashSet(Vertex)
        preorder_components = VectorArray(100, VectorArray)
        postorder_components = VectorArray(100, VectorArray)

        for neighbour in self.obj.vertices():
            # skip vertex if visited already.
            if neighbour in visited: continue
            # utilizes our single component version to get both pre and post order results.
            preorder, postorder = self.dfs_combined_iterative_traversal(neighbour)
            # mark as visited. (postorder not necessary -- Every node that appears in postorder already appeared in preorder.)
            for i in preorder: visited.add(i)
            # append to components arrays.
            preorder_components.append(preorder)
            postorder_components.append(postorder)
        return preorder_components, postorder_components

    def bfs_forest(self):
        """
        Operates on the entire graph, not just one component.
        No single start vertex, Iterates BFS over all unvisited vertices
        Required for disconnected graphs
        This is a Connected Components Algorithm in practice.
        """
        visited = HashSet(Vertex)
        levelorder_components = VectorArray(100, VectorArray)

        for neighbour in self.obj.vertices():
            if neighbour in visited: 
                continue
            levelorder = self.bfs_iterative_traversal(neighbour)
            for i in levelorder:
                visited.add(i)
            levelorder_components.append(levelorder)
        return levelorder_components





