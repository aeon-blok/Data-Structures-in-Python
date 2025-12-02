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
import copy
from pprint import pprint

# endregion

# region custom imports
from user_defined_types.generic_types import (
    T,
    K,
    ValidDatatype,
    TypeSafeElement,
    Index,
    ValidIndex,
)
from user_defined_types.key_types import iKey, Key
from utils.validation_utils import DsValidation
from utils.representations import DisjointSetForestRepr
from utils.helpers import RandomClass
from utils.exceptions import *

from adts.collection_adt import CollectionADT
from adts.disjoint_set_adt import DisjointSetADT

from ds.sequences.Stacks.array_stack import ArrayStack
from ds.sequences.Deques.linked_list_deque import DllDeque
from ds.primitives.arrays.dynamic_array import VectorArray, VectorView
from ds.maps.hash_table_with_open_addressing import HashTableOA
from ds.trees.tree_nodes import AncestorRankNode

# endregion


class DisjointSetForest(DisjointSetADT[T], CollectionADT[T]):
    """
    Standard Union Find Data structure with Ancestor Nodes and Parent Pointer Tree structure.
    Utilizes Path Compression
    Utilizes Union By Rank
    O(1) - time complexity for standard operations
    """
    def __init__(self, datatype: type) -> None:
        self._datatype = ValidDatatype(datatype)
        self._nodes = HashTableOA(AncestorRankNode)

        # composed objects
        self._desc = DisjointSetForestRepr(self)

    @property
    def datatype(self):
        return self._datatype

    @property
    def representatives(self):
        """returns a set of all the representatives in the Disjoint set forest"""

        roots = set()   # todo change to custom set later

        # * collect roots from hash table.
        for node in self._nodes.values():
            if node.parent == node:
                root = self.find_representative(node.element)
                roots.add(root)
        return roots

    # ----- Meta Collection ADT Operations -----
    def is_empty(self) -> bool:
        return len(self._nodes) == 0

    def clear(self) -> None:
        self._nodes.clear()

    def __contains__(self, element: T) -> bool:
        element = TypeSafeElement(element, self.datatype)
        return id(element) in self._nodes

    def __iter__(self):
        for rep in self.representatives: yield rep

    def __len__(self) -> Index:
        return len(self._nodes)

    # ----- Utilities -----

    def create_children_index(self):
        """recreates the children attribute for parent pointer nodes"""
        # * intialize storage
        # key = parent, value = stack of children
        parent_child_index = HashTableOA(ArrayStack, capacity=100)
        # * we need to know for each node: who is the parent?
        for node in self._nodes.values():

            # * path compression
            self.find_representative(node.element)

            # initialize parent
            parent = node.parent
            parent_key = str(parent.element)

            # * representative case: node is a root not a child.
            if parent == node:
                continue
            # *Existing Parent Case: does a key already exist in the hashtable?
            # then access it and add this node to the list of children.
            children_stack = parent_child_index.get(parent_key)
            if children_stack is not None:
                children_stack.push(node)
                parent_child_index.put(parent_key, children_stack)
            # * New Parent Case: if a key does not exist. create a new list and add this node
            else:
                children_stack = ArrayStack(AncestorRankNode)
                children_stack.push(node)
                # add to the hash table.
                parent_child_index.put(parent_key, children_stack)
        return parent_child_index


    def get_members(self, representative: T) -> Optional[ArrayStack[AncestorRankNode[T]]]:
        """
        retrieves all the member NODES from a representative and returns a stack of them... 
        takes a representative element value as input (easy to search for...)
        """
        rep_node = self.find_representative(representative)

        if rep_node is None:
            raise NodeEmptyError(f"Error: Node cannot be None.")
        if rep_node not in self.representatives:
            raise NodeExistenceError(f"Error: representative: {rep_node.element} was not found. please check the representatives list.")

        # * collect all members of a set - starting from representative
        set_members = ArrayStack(AncestorRankNode, 100)
        # if the parent is the representative - add to stack
        for node in self._nodes.values():
            if node.parent == rep_node and node != rep_node:
                set_members.push(node)
        return set_members

    def visualize_representative(self, representative: T):
        """Visualizes the inner parent pointer tree of a single represenatitve"""
        # * find and validate the representative (node)
        rep_node = self.find_representative(representative)
        set_members = self.get_members(representative)
        bush_structure = ""

        # validation
        if rep_node is None:
            raise NodeEmptyError(f"Error: Node cannot be None.")
        if rep_node not in self.representatives:
            raise NodeExistenceError(f"Error: representative: {rep_node.element} was not found. please check the representatives list.")

        rep_key = str(rep_node.element)
        rep_string = f"[🏛️  rep: {rep_node.element}[r={rep_node.rank}]]"
        title = f"🌴 Disjoint Set: Parent Pointer Tree:"
        generate_members = f', '.join(i.element for i in set_members) if set_members else None
        members_string = f"[set_members={generate_members}]"

        # * get the children of our representative
        # initialize parent-child hashtable - key = parent, value=stack of children.
        parent_child_index = self.create_children_index()
        children_stack = parent_child_index.get(rep_key)
        tree_size: int = 1
        tree_children = parent_child_index.get(rep_key)
        child_strings_stack = ArrayStack(str, 100)

        while tree_children:
            child_node = tree_children.pop()
            child_string = f" └─ {child_node.element}"
            child_strings_stack.push(child_string)
            tree_size +=1

        # * empty tree - just the representative.
        if tree_size == 1:
            return f"\n{title}\n{rep_string}"

        # * main case: return final bush construction
        tree_size_string = f"[tree_size={tree_size}]"
        while child_strings_stack:
            bush_structure += child_strings_stack.pop() + "\n"
        stats = f"{tree_size_string}{members_string}"
        return f"\n{title}\n{stats}\n{rep_string}\n{bush_structure}\n"

    def __str__(self) -> str:
        return self._desc.str_disjoint_set_forest()

    def __repr__(self) -> str:
        return self._desc.repr_disjoint_set_forest()

    # ----- Canonical ADT Operations -----
    def make_set(self, element: T) -> None:
        """Creates a new disjoint set (a parent pointer tree) - the input element becomes the representative (or root) of this set."""
        element = TypeSafeElement(element, self.datatype)
        key = str(element)

        # * already exists case: check if element already exists in a set - if so return its representative. (via find operation)
        if key in self._nodes:
            existing_node = self._nodes.get(key)
            raise NodeExistenceError(f"Error: Node already exists in disjoint set. Representative: {self.find(existing_node.element)}")

        # input the node as the element.
        node = AncestorRankNode(self.datatype, element)
        self._nodes.put(key, node)
        node.increment_rank # increment rank attribute for the node.

    def find_representative(self, element: T) -> Optional[AncestorRankNode[T]]:
        """Recursive helper method that finds the root node of a set with path compression"""
        element = TypeSafeElement(element, self.datatype)
        key = str(element)
        node = self._nodes.get(key)
        # why can we no longer find the key after a rehash.
        if node is None:
            raise NodeExistenceError(f"Error: Element does not exist in Disjoint Set Forest...")

        # * here is where we find the representative.
        if node != node.parent:
            # * path compression logic -- changes the parent of the node to point to the root. recursively repeats for all nodes traversed until the root.
            # future find calls on any node on that path are O(1).
            # recursively traverses up the tree until it finds the root.
            node.parent = self.find_representative(node.parent.element)
            return node.parent

        # * node is the root / representative case
        elif node == node.parent:
            return node

    def find(self, element: T) -> Optional[T]:
        """
        returns the representative (root) element value of the set that the input element is a part of.
        public wrapper for recursive helper method with path compression for O(1) - lookups
        """
        return self.find_representative(element).element

    def union(self, element_x: T, element_y: T) -> bool:
        """
        we search for 2 elements - x and y.
        if the elements are aready in the same set we return false.
        We compare the size each set.
        we merge the two sets together.
        using the rank attribute - we attach the smaller set as a child of the larger set. in essence - the smaller rank set, becomes a subtree of the larger rank set.
        if the two sets are the same rank - parent y to x
        """

        element_x = TypeSafeElement(element_x, self.datatype)
        element_y = TypeSafeElement(element_y, self.datatype)
        key_x = str(element_x)
        key_y = str(element_y)

        # * find the representative for each element -- its a recursive method
        root_x = self.find_representative(element_x)
        root_y = self.find_representative(element_y)

        # * Same Set Case: elements exist in the same set. (cannot perform union)
        if root_x == root_y:
            return False

        # * Union By Rank: attack the smaller set to the bigger set.
        if root_x.rank < root_y.rank:
            root_x.parent = root_y
        elif root_x.rank > root_y.rank:
            root_y.parent = root_x
        else:
            # only increment by 1 when the heights are equal
            root_y.parent = root_x
            root_x.increment_rank

        # * Update representatives list via path compression.
        # we need to find the element in order to update the implicit tree pointers.
        # this will allow us to see the merged results in our updated total sets count and in str etc
        self.find_representative(element_x)
        self.find_representative(element_y)
        return True

    def set_count(self) -> int:
        """ 
        counts the number of disjoint sets 
        by recursively finding the root of each element in the hash table. 
        then adding the root to a set() (no duplicates).
        """
        representatives = set() # todo change to custom set later
        for node in self._nodes.values():
            if node.parent == node:
                root_node = self.find_representative(node.element)
                representatives.add(root_node)
        return len(representatives)


# --------- Main Client Facing Code -----------
def main():

    string_data = [
        "apple",
        "banana",
        "cherry",
        "date",
        "elderberry",
        "fig",
        "grape",
        "honeydew",
        "kiwi",
        "lemon",
        "mango",
        "nectarine",
        "orange",
        "papaya",
        "quince",
        "raspberry",
        "strawberry",
        "tangerine",
        "ugli",
        "watermelon",
    ]

    dsf = DisjointSetForest(str)
    print(dsf)
    print(repr(dsf))

    for i in string_data:
        key = Key(str(i))
        dsf.make_set(i)

    print(dsf)
    print(repr(dsf))

    def test_union():
        """tests the union method for disjoint sets. with helpful debugging strings."""
        reps = list(dsf.representatives)
        x = random.choice(reps).element
        y = random.choice(reps).element
        result = dsf.union(x, y)
        print(f"\nTesting Union: x={x} & y={y}, (Success?: {result}) Set Count after op: {dsf.set_count()}")
        print(f"{', '.join(f'{i.element}[r={i.rank}]' for i in dsf.representatives)}")

    for i in range(17):
        test_union()

    def test_find():
        """testing the find functionality of disjoint set forest...."""
        items = dsf._nodes.values()
        x = random.choice(items).element
        node = dsf.find_representative(x)
        print(f"\nTesting Find: {x}, representative (parent)={node.parent.element}")
        result = dsf.find(x)
        print(f"result: {result}")
        # print(f"Testing Find with non existent element")
        # try:
        #     x = "NIL"
        #     result = dsf.find(x)
        # except Exception as e:
        #     print(f"{e}: element={x}")

    for i in range(10):
        test_find()

    print(dsf)
    print(repr(dsf))
    print(f"Elements in Hash Table: {', '.join(i.element for i in dsf._nodes.values())}")

    print(f"\nTesting visualization of a representative: ")
    rep = random.choice(list(dsf.representatives)).element
    print(f"representative={rep}")
    members = dsf.get_members(rep)
    members_string = f', '.join(i.element for i in members) if members else None
    print(f"members={members_string}")
    print(f"{dsf.visualize_representative(rep)}")


if __name__ == "__main__":
    main()
