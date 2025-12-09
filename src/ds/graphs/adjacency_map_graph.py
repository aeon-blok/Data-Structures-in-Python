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
from utils.representations import GraphRepr
from utils.exceptions import *

from adts.collection_adt import CollectionADT
from adts.graph_adt import GraphADT

from user_defined_types.generic_types import (
    ValidDatatype,
    TypeSafeElement,
    PositiveNumber,
)
from user_defined_types.key_types import iKey, Key
from user_defined_types.graph_types import VertexColor, ValidVertex, weight

from ds.primitives.arrays.dynamic_array import VectorArray, VectorView
from ds.primitives.Linked_Lists.sll import Sll_Node, LinkedList
from ds.primitives.Linked_Lists.dll import Dll_Node, DoublyLinkedList
from ds.primitives.Linked_Lists.dcll import DoublyCircularList
from ds.sequences.Stacks.array_stack import ArrayStack
from ds.sequences.Queue.linked_list_queue import LlQueue
from ds.sequences.Deques.linked_list_deque import DllDeque
from ds.maps.hash_table_with_chaining import ChainHashTable
from ds.maps.hash_table_with_open_addressing import HashTableOA
from ds.maps.Sets.hash_set import HashSet
from ds.trees.Priority_Queues.binary_heap import BinaryHeap
from ds.trees.Disjoint_Sets.disjoint_set_forest import DisjointSetForest, AncestorRankNode
from ds.graphs.vertex import Vertex
from ds.graphs.edge import Edge

from ds.graphs.graph_utils import GraphUtils

# endregion

# todo create type alias for hash tables so we can swap the representation easily without breaking things.

class GraphAdjMap(GraphADT[T], CollectionADT[T], Generic[T]):
    """
    Graph: Data Structure Implementation
    Simple Graph Implementation.
    Represented via Adjacency Map (hashtable)
    Utilizes Vertex Node Objects and Edge Objects to ensure consistency...
    """

    def __init__(self, datatype: type, directed: bool = False, edge_weight_datatype: type = float) -> None:
        super().__init__()
        self._datatype = ValidDatatype(datatype)
        self._edge_weight_datatype = edge_weight_datatype
        # * Adjacency Map representation - they store two different views of the same collection of edges and verts
        # structure = Level 1 Map: Key: Vertex obj, Value: Map of neighbour verts
        # Level 2 Map: Key: Vertex_Neighbour_obj, Value: edge_obj(between vertex and vertex neighbour)
        # Directed: out = edges directed outwards, in = edges directed inwards.
        # Undirected: Each edge is stored symmetrically in both maps:
        self._out_adj_map: dict[Vertex, dict[Vertex, Edge]] = ChainHashTable(ChainHashTable)
        self._inc_adj_map: dict[Vertex, dict[Vertex, Edge]] = ChainHashTable(ChainHashTable) if directed else self._out_adj_map
        self._inserted_edges_counter = 0    # only tracks the number of edges inserted. (doesnt decrease)
        self._inserted_vertex_counter = 0 # only tracks the number of verts inserted. (doesnt decrease)

        # composed objects
        self._utils = GraphUtils(self)
        self._desc = GraphRepr(self)

    @property
    def datatype(self) -> type:
        return self._datatype

    @property
    def is_directed(self) -> bool:
        """returns true if graph is directed."""
        return self._inc_adj_map is not self._out_adj_map

    @property
    def vertex_count(self) -> int:
        """returns the total number of vertices in the graph..."""
        return len(self._out_adj_map)

    @property
    def edge_count(self) -> int:
        """returns the number of edges in the graph. undirected edges will be 1/2 of the directed edges..."""
        directed_total = sum(len(self._out_adj_map[vertex]) for vertex in self._out_adj_map)
        undirected_total = directed_total // 2
        return directed_total if self.is_directed else undirected_total

    @property
    def view_adjacency_map(self) -> str:
        return self._utils.view_adjacency_map()

    # ----- Utility Operations -----
    def __str__(self) -> str:
        return self._desc.str_graph()

    def __repr__(self) -> str:
        return self._desc.repr_graph()

    # ----- Canonical ADT Operations -----
    def has_vertex(self, vertex: Vertex) -> bool:
        """returns a boolean checking for existence of a vertex"""
        return vertex in self._out_adj_map

    def has_edge(self, u: Vertex, v: Vertex) -> bool:
        """checks if an edge exists and returns true or false."""
        # vertices dont exist.
        if u not in self._out_adj_map or v not in self._out_adj_map:
            return False
        # directed graph - only check u -> v
        if self.is_directed:
            return v in self._out_adj_map[u]
        # undirected graph: check both directions...
        return v in self._out_adj_map[u] and u in self._out_adj_map[v]

    # ----- Accessors -----
    def vertices(self, return_element: bool = False):
        """returns vertex objects / elements from the graph. Can utilize an Array or a generator."""
        verts = VectorArray(self.vertex_count * 2, Vertex)
        for vertex in self._out_adj_map.keys():
            vertex = vertex.element if return_element else vertex
            verts.append(vertex)
        return verts

    def edges(self, return_element: bool = False):
        """yield edge objects / elements from the graph"""
        results = HashSet(Edge)
        for neighbours_map in self._out_adj_map.values():
            for edge in neighbours_map.values():
                edge = edge.element if return_element else edge
                results.add(edge)
        return results.members

    def neighbours(self, vertex, outgoing=True, return_element: bool = False):
        """returns the neighbours for the specified vertex. either as a generator or an array"""
        neighbours_map = self._out_adj_map if outgoing else self._inc_adj_map
        if vertex not in neighbours_map:
            raise NodeExistenceError(f"Error: Neighbour Vertex does not exist in this vertex's neighbours map.")
        # array
        vert_neighbours = VectorArray(self.vertex_count * 2, Vertex)

        for neighbour in neighbours_map[vertex].keys():
            neighbour = neighbour.element if return_element else neighbour
            vert_neighbours.append(neighbour)

        return vert_neighbours

    def degree(self, vertex: Vertex, outgoing=True) -> int:
        """returns the degree (number of edges) of the specified vertex"""
        vertex = ValidVertex(vertex, Vertex)
        neighbours_map = self._out_adj_map[vertex] if outgoing else self._inc_adj_map[vertex]
        return len(neighbours_map)

    def get_edge(self, u: Vertex, v: Vertex):
        """gets the edge from U -> V specified in the inputs."""
        # * guard clause: Existence check
        if not self.has_edge(u,v):
            raise NodeExistenceError(f"Error: Couldnt Locate this Edge in the graph.")
        return self._out_adj_map[u][v]

    def incident_edges(self, vertex: Vertex, return_element=False):
        """returns the incident edges for a specified vertex"""
        # existence check:
        if vertex not in self._out_adj_map:
            raise NodeExistenceError(f"Error: Vertex does not exist in the graph.")

        # stores edges
        edges = VectorArray(self.edge_count * 2, Edge)
        # set to prevent duplicates
        seen = HashSet(Edge)

        # outgoing edges
        for edge in self._out_adj_map[vertex].values():
            if edge not in seen:
                edge = edge.element if return_element else edge
                seen.add(edge)
                edges.append(edge)

        # incoming edges: (only if graph is directed.)
        if self.is_directed:
            for edge in self._inc_adj_map[vertex].values():
                if edge not in seen:
                    edge = edge.element if return_element else edge
                    seen.add(edge)
                    edges.append(edge)

        return edges

    # ----- Mutators -----
    def add_edge(self, u, v, element: weight | None = None) -> Edge:
        """Adds an edge between two vertices and returns the Edge object."""

        # * Check do vertices exist?
        if self._out_adj_map.get(u) is None:
            raise NodeExistenceError(f"Error: Vertex does not exist in the graph.")
        if self._out_adj_map.get(v) is None:
            raise NodeExistenceError(f"Error: Vertex does not exist in the graph.")

        # * initialize edge (will validate element and vertices - but not none.)
        edge = Edge(self._edge_weight_datatype, u, v, element)
        self._out_adj_map[u][v] = edge
        self._inc_adj_map[v][u] = edge   

        # * increment edge insertion order counter and assign to new edge
        self._inserted_edges_counter +=1
        edge.insert_order = self._inserted_edges_counter

        return edge

    def add_vertex(self, element, label=None, vertex_comparison_key=None) -> Vertex:
        """Adds a Vertex to the Graph and returns the vertex object for use as a reference key"""

        # todo the first comparison key entered becomes the TABLE comparison key,
        # todo all other comparison keys must match this key or an error is raised.

        # * initialize Vertex Object
        vertex = Vertex(self.datatype, element, label, vertex_comparison_key)
        # * Initialize Neighbours Map
        self._out_adj_map[vertex] = ChainHashTable(Edge)
        # For directed graphs
        if self.is_directed: self._inc_adj_map[vertex] = ChainHashTable(Edge)

        # * increment vertex insertion order counter and assign to new vertex
        self._inserted_vertex_counter += 1
        vertex.insert_order = self._inserted_vertex_counter

        return vertex

    def remove_edge(self, u, v) -> None:
        """removes an edge from the graph."""

        # * guard clause: edge not found
        if not self.has_edge(u,v):
            raise NodeExistenceError(f"Error: Edge does not exist in the graph.")

        # * delete edges
        # directed graph:
        if self.is_directed:
            # Both entries refer to the same Edge object, represented in the 2 adjacency maps.
            del self._out_adj_map[u][v]
            del self._inc_adj_map[v][u]

        # undirected graph:
        else:
            # one logical edge (u, v) - for undirected the incoming map is a reference to the outgoing.
            del self._out_adj_map[u][v]  
            # extra logic for self loops... ([u][u] etc) -- if u == v (then u IS v), its already been deleted.
            if u != v: del self._out_adj_map[v][u]

    def remove_vertex(self, vertex) -> None:
        """removes a Vertex and all its incident Edges from the graph."""
        # * validate input
        vertex = ValidVertex(vertex, Vertex)

        # * guard clause: vertex doesnt exist in the graph.
        if not self.has_vertex(vertex):
            raise NodeExistenceError(f"Error: Vertex does not exist in the graph.")

        # * 1.) first delete edges
        out_neighbours = self._out_adj_map[vertex].keys()
        inc_neighbours = self._inc_adj_map[vertex].keys()
        # directed graph - remove from outgoing and incomming adjacency maps.
        if self.is_directed:
            for neighbour in out_neighbours:
                self.remove_edge(vertex, neighbour)
            # for incoming edges, we remove from the neighbour -> vertex
            for neighbour in inc_neighbours:
                self.remove_edge(neighbour, vertex)
        else:
            # undirected graph - just remove from outgoing adjacency map.
            for neighbour in out_neighbours:
                self.remove_edge(vertex, neighbour)

        # * 2.) delete the vertex object
        # undirected graph: delete from outgoing adjacency map only
        del self._out_adj_map[vertex]
        # directed graph: delete from incoming adjacency map also.
        if self.is_directed: del self._inc_adj_map[vertex]

    # ----- Traversals -----
    def dfs_forest(self):
        """
        returns both preorder and postorder arrays of a dfs search.
        utilizes iterative traversal. (its also a connected components algorithm)
        returns an MD array of component graphs. for both preorder and postorder traversal.
        """
        preorder, postorder = self._utils.dfs_forest()
        return preorder, postorder

    def bfs_forest(self):
        """
        Breadth First Search via iterative traversal & deque (a connected components algorithm)
        Will iterate through all component graphs and return the results as a MD array.
        """
        return self._utils.bfs_forest()


# ------------------------ Main(): Client Facing Code --------------------------
def main():

    input_data_a = ["A", "B", "C", "D", "E","F", "G", "H", "I", "J"]
    input_data_b = [
        "K",
        "L",
        "M",
        "N",
        "O",
        "P",
        "Q",
        "R",
        "S",
        "T",
        "U",
        "V",
        "W",
        "X",
        "Y",
        "Z"
    ]

    print(f"Testing Undirected Graph: str")
    g = GraphAdjMap[str](str)
    print(repr(g))
    print(g)

    print(f"\nAdding Vertices to the graph.... (add edges also...)")
    for i in input_data_a:
        neighbour = f"New_{i}"
        label = f"tagged"
        random_weight = round(random.random(), 2)
        vert_a = g.add_vertex(i)
        vert_b = g.add_vertex(neighbour, label)
        g.add_edge(vert_a, vert_b, random_weight)
    print(repr(g))
    print(g)

    print(f"\nTesting iteration over vertices and edges")
    print(repr(g))
    current_verts = g.vertices()
    print(f"{current_verts}")
    current_edges = g.edges()
    print(current_edges)

    print(f"\nTesting finding a random edge...")
    random_vert_z = random.choice(current_verts)
    random_vert_y = random.choice(current_verts)
    print(f"Does Edge Exist? between {random_vert_z} and {random_vert_y}: {g.has_edge(random_vert_z, random_vert_y)}")
    print(f"Get neighbours from {random_vert_y}: {g.neighbours(random_vert_y)}")
    y_neighbour = g.neighbours(random_vert_y)
    print(f"Does Edge Exist? between {random_vert_y} and {y_neighbour[0]}: {g.has_edge(random_vert_y, y_neighbour[0])}")
    print(f"Get the edge from these two vertices.")
    y_edge = g.get_edge(random_vert_y, y_neighbour[0])
    print(f"Edge = {y_edge}")

    print(f"\nTesting remove vertex functionality...")
    target_for_del = random.choice(current_verts)
    print(f"target: {target_for_del} at {repr(target_for_del)}. Degree: {g.degree(target_for_del)}")
    g.remove_vertex(target_for_del)
    print(repr(g))
    print(g)

    print(f"\nTesting Remove Edge Functionality... (and neighbours functionality)")
    new_verts = [v for v in g.vertices() if g.degree(v) > 0]
    random_vert_a = random.choice(new_verts)
    his_neighbours = g.neighbours(random_vert_a)
    print(f"Incident edges BEFORE removal. {random_vert_a}: {g.incident_edges(random_vert_a)}")
    random_vert_b = his_neighbours[0]
    print(f"Targeting:")
    print(f"{repr(random_vert_a)}<->{repr(random_vert_b)}")
    g.remove_edge(random_vert_a, random_vert_b)
    print(repr(g))
    print(g)
    print(f"Incident edges after removal. {random_vert_a}: {g.incident_edges(random_vert_a)}")

    # 1. HIGH-DENSITY CONNECTIVITY TEST (6–10 edges per vertex)
    print(f"\nTesting high-density connectivity (6–10 edges per vertex)...")
    g_dense = GraphAdjMap[str](str)

    # add vertex and append to a list at the same time....
    dense_vertices = []
    for name in input_data_b:
        dense_vertices.append(g_dense.add_vertex(name))

    # connect each vertex to up to 6–10 others
    for i, v in enumerate(dense_vertices):
        neighbours = dense_vertices[:i] + dense_vertices[i + 1 :]
        random.shuffle(neighbours)

        edge_count = random.randint(6, min(10, len(neighbours)))
        for n in neighbours[:edge_count]:
            if not g_dense.has_edge(v, n):
                g_dense.add_edge(v, n, round(random.random(), 3))

    print(repr(g_dense))
    print(g_dense)

    # sanity checks
    print("\nDegree sanity check:")
    for v in g_dense.vertices():
        deg = g_dense.degree(v)
        print(f"{repr(v)} degree = {deg}")
        assert deg >= 6, "Degree invariant violated"

    # 2. TYPE-SAFETY TESTS
    print("\nTesting type safety...")

    try:
        print("Attempting to add vertex with wrong type (int)...")
        g_dense.add_vertex(123)
    except Exception as e:
        print(f"Caught expected error: {type(e).__name__} -> {e}")

    try:
        print("Attempting to add edge with raw value instead of Vertex...")
        g_dense.add_edge("apple", "banana", 0.5)
    except Exception as e:
        print(f"Caught expected error: {type(e).__name__} -> {e}")

    try:
        print("Attempting to check edge using mismatched vertex types...")
        fake_vert = g_dense.add_vertex("new_vertex")
        g_dense.has_edge(fake_vert, "banana")
    except Exception as e:
        print(f"Caught expected error: {type(e).__name__} -> {e}")
    finally:
        g_dense.remove_vertex(fake_vert)

    # 3. NON-EXISTENT VERTEX / EDGE TESTS
    print("\nTesting non-existent vertex / edge behavior...")
    orphan_vertex = Vertex(str, "orphan")  # not added

    try:
        print("Calling neighbours() on non-existent vertex...")
        g_dense.neighbours(orphan_vertex)
    except Exception as e:
        print(f"Caught expected error: {type(e).__name__} -> {e}")

    try:
        print("Removing non-existent vertex...")
        g_dense.remove_vertex(orphan_vertex)
    except Exception as e:
        print(f"Caught expected error: {type(e).__name__} -> {e}")

    verts = g_dense.vertices()
    v1 = verts[0]
    v1_neighbours = g_dense.neighbours(v1)

    print(f"{v1} & {v1_neighbours}")
    print(verts)
    for v in v1_neighbours:
        v_edge = g_dense.get_edge(v1, v)
        # print(f"Removing valid edge...{v_edge}")
        g_dense.remove_edge(v1, v)

    print(f"{v1}: neighbours: {g_dense.neighbours(v1)}")
    v2 = v1_neighbours[0]
    try:
        print("Removing non existent edge...")
        g_dense.remove_edge(v1,v2)
    except Exception as e:
        print(f"Caught expected error: {type(e).__name__} -> {e}")

    # 5. SELF-LOOP TEST
    print("\nTesting self-loop behavior...")
    dense_vertices = g_dense.vertices()
    print(dense_vertices)
    try:
        self_v = dense_vertices[0]
        print(f"Attempting self-loop on {repr(self_v)}")
        g_dense.add_edge(self_v, self_v, 0.42)
        print("Self-loop accepted.")
        print(f"Incident edges: {g_dense.incident_edges(self_v)}")
    except Exception as e:
        print(f"Self-loop rejected as expected: {type(e).__name__} -> {e}")

    print(f"\nTesting DFS: Depth First Search")
    print(f"Current Verts: {g_dense.vertices()}")
    for i in g_dense.vertices():
        print(f"{i}: {g_dense.neighbours(i)}")
    preforest, postforest = g_dense.dfs_forest()
    print(f"Preorder Component Graph Forest:")
    print(preforest)

    print(f"\nTesting BFS: Breadth First Search")
    print(f"Current Verts: {g_dense.vertices()}")
    for i in g_dense.vertices():
        print(f"{i}: {g_dense.neighbours(i)}")
    levelforest = g_dense.bfs_forest()
    print("BFS Levelorder Component Graph Forest: ")
    print(levelforest)


if __name__ == "__main__":
    main()
