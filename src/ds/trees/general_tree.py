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
from pprint import pprint
# endregion

# region custom imports
from utils.custom_types import T
from utils.validation_utils import DsValidation
from utils.representations import GenTreeRepr
from utils.helpers import RandomClass
from utils.exceptions import *

from adts.collection_adt import CollectionADT
from adts.tree_adt import TreeADT, iTNode

from ds.sequences.Stacks.array_stack import ArrayStack
from ds.primitives.arrays.dynamic_array import VectorArray, VectorView
from ds.trees.tree_nodes import TNode
from ds.trees.tree_utils import TreeUtils


# endregion

"""
General Tree Implementation: N-ary Tree.
using nodes.
"""


# Implementation:
class GeneralTree(TreeADT[T]):
    """
    Tree Data Structure: Allows for 3 Traversal types (DFS, reverse DFS & BFS)
    Tree Nodes can create their own children and become subtrees.
    """
    def __init__(
            self,
            datatype: type, 
            iteration_type: Literal['pre order', 'post order', 'level order']='pre order',
            ) -> None:

        self._root: Optional[iTNode[T]] = None
        self.iteration_type = iteration_type
        self._datatype = datatype

        # composed objects
        self._utils = TreeUtils(self)
        self._validators = DsValidation()
        self._desc = GenTreeRepr(self)

        self._validators.validate_datatype(self._datatype)

    @property
    def datatype(self):
        return self._datatype

    # ----- Utilities -----
    def view(self):
        """
        Traverses the Tree via stack
        adds connector symbols in front of each node value, depending on whether it is the last child "└─" or one of many "├─",
        every node adds either " " if parent is last child (no vertical bar needed) or "| " if parent is not last child (vertical bar continues)
        the node & its display symbols are appended to a list for the final string output.
        """
        if self.root is None:
            return f"< 🌳 empty tree>"

        hierarchy = []
        tree = [(self.root, "", True)]  # (node, prefix, is_last)

        while tree:
            # we traverse depth-first, which naturally fits a hierarchical print.
            node, prefix, is_last = tree.pop()

            # root (depth = 0), we print 🌲
            if node is self.root:
                indicator = "🌲:"
            # decides what connector symbol appears before the node value when printing the tree.
            else: 
                indicator = "" if prefix == "" else ("└─" if is_last else "├─")

            # add to final string output
            hierarchy.append(f"{prefix}{indicator}{str(node.value)}") 

            # Build prefix for children - Vertical bars "│" are inherited from ancestors that are not last children
            new_prefix = prefix + ("   " if is_last else "│  ")

            # Iterates over the node’s children in reverse. (left to right) --- enumerate gives index i for calculating new prefix.
            for i, child in enumerate(reversed(node.children)):
                last_child = (i==0)
                # Update ancestor flags: current node's is_last boolean affects all its children
                tree.append((child, new_prefix, last_child))    

        node_structure = "\n".join(hierarchy)

        return f"\nTree Total Nodes: {len(self)}\n{node_structure}\n"

    def flattened_view(self):
        # utilizes __iter__ which has 3 different traversal algos
        node_values = [node for node in self]    
        return f"[{', '.join(node_values)}]"

    def bfs_view(self):
        return self._utils.view_bfs()

    def __str__(self) -> str:
        return self._desc.str_gen_tree()

    def __repr__(self) -> str:
        return self._desc.repr_gen_tree()

    # ----- Canonical ADT Operations -----
    # ----- Accessors -----
    @property
    def root(self):
        """returns the root node"""
        return self._root

    def parent(self, node):
        """returns the parent NODE of a specified node"""
        self._utils.validate_tree_node(node)
        return node.parent

    def children(self, node):
        """returns a list of all children nodes of a specified node"""
        self._utils.validate_tree_node(node)
        return node.children

    def num_children(self, node):
        """returns the total number of children of a specified node"""
        self._utils.validate_tree_node(node)
        return len(node.children)

    def is_root(self, node):
        """returns true if the node is the root of a tree"""
        self._utils.validate_tree_node(node)
        return node == self._root

    def is_leaf(self, node):
        """returns True if the node is a leaf node (no children)"""
        self._utils.validate_tree_node(node)
        return len(node.children) == 0

    def is_internal(self, node):
        """returns True if the node has children nodes."""
        self._utils.validate_tree_node(node)
        return len(node.children) > 0

    def depth(self, node):
        self._utils.validate_tree_node(node)
        return self._utils.tree_depth(node)

    def height(self, node):
        self._utils.validate_tree_node(node)
        return self._utils.tree_height(node)

    # ----- Mutators -----
    def createTree(self, value):
        """creates a new tree with a root node"""
        self._validators.enforce_type(value, self._datatype)
        self._root = TNode(self._datatype, value, tree_owner=self)
        return self._root

    def addChild(self, parent_node, value):
        """adds a child node to the specified node."""
        self._validators.enforce_type(value, self._datatype)
        child = TNode(self._datatype, value, tree_owner=self)
        child.parent = parent_node  # link to parent.
        parent_node.children.append(child) # link  parent to child.
        return child

    def remove(self, node):
        """
        removes a specified node and all its descendants
        """

        self._utils.validate_tree_node(node)

        # 1. Store Node & Subtree -- Capture subtree size before modifying anything
        deleted_node = node  # store node to return later

        # 2. Unlink from parent (remove from children list) BEFORE deleting parent pointers
        parent = node.parent
        node.tree_owner = None
        node.deleted = True
        if parent is not None:
            parent.children.remove(node)

        # 3. Iteratively dereference Node & subtree using stack
        subtree = [node]    # note its the actual node input not a variable.
        while subtree:
            node = subtree.pop()
            subtree.extend(node.children)
            node.children = []  # empties list of children
            # dereferences parent node so it no longer points to the node. (becomes a leaf node)
            node.parent = None
            node.tree_owner = None
            node.deleted = True
        return deleted_node

    def replace(self, node, value):
        """replaces a value in a specified node. Does NOT replace the subtree. Structure remains the same."""
        replace_node = node
        replace_node.value = value
        return replace_node

    # ----- Traversals -----
    def preorder(self):
        """Depth First Search: (DFS)"""
        return [i for i in self._utils.dfs_depth_first_search()]

    def postorder(self):
        """Reversed Depth First Search: (RDFS) travels from last child to root - returns a list of values"""
        return [i for i in self._utils.reverse_dfs_postorder_search()]

    def level_order(self):
        """Breadth First Search: (BFS) -- traverses the tree horizontally a level at a time."""
        return [i for i in self._utils.bfs_breadth_first_search()]

    # ----- Meta Collection ADT Operations -----
    def is_empty(self):
        return self.root is None

    def __len__(self):
        """returns total number of nodes in the tree"""
        return self._utils.count_total_tree_nodes()

    def clear(self):
        """The tree is completely emptied and all children nodes are dereferenced."""
        tree = [self.root]
        while tree:
            node = tree.pop()
            tree.extend(node.children)  # add children to the processing line.
            # dereference node
            node.children = []
            node.parent = None
        self.root = None

    def __contains__(self, value):
        """checks to see if any nodes in the tree contain the value."""
        if not self.root:
            return False
        tree = [self.root]  # change to custom stack later....
        while tree:
            node = tree.pop()
            if node.value == value:
                return True
            tree.extend(reversed(node.children))
        return False

    def __iter__(self):
        """iterates over the tree via 3 traversal methods, DFS, DFS reversed & BFS"""
        if self.iteration_type == 'pre order':
            return self._utils.dfs_depth_first_search()
        elif self.iteration_type == 'post order':
            return self._utils.reverse_dfs_postorder_search()
        elif self.iteration_type == 'level order':
            return self._utils.bfs_breadth_first_search()
        else:
            raise KeyInvalidError(f"Error: Iteration Type: {self.iteration_type} is Invalid.")


# Main ---- Client Facing Code


# todo create a size counter for tree. will mean O(1) for __len__ instead of O(N) - however delete will be O(H)
# todo test adding deleted nodes to the tree - should be error
# todo more testing required.

def main():
    # -------------- Testing Tree Functionality -----------------
    tree = GeneralTree[str](str, iteration_type="pre order")
    print(repr(tree))
    # print(f"Testing is_empty: {tree.is_empty()}")
    root = tree.createTree("wales")

    # print(f"Adding Child to Tree:")
    child_a = tree.addChild(root, "a child of summer")
    child_b = tree.addChild(child_a, "a child of winter")
    # print(f"^ root children: {root.children}")
    # print(f"^ children: {child_a.children}")
    # print(f"^ children: {child_b.children}")

    print(tree)

    child_c = tree.addChild(child_a, "a child of spring")
    child_d = tree.addChild(root, "a child of autumn")
    child_dd = tree.addChild(child_d, "fall colors")
    child_de = tree.addChild(child_dd, "fall wind")
    child_e = tree.addChild(root, "ttettttst")
    print(f"Testing is_empty: {tree.is_empty()}")
    print(tree)
    print(tree.bfs_view())
    print(f"removing Children from Tree:")
    tree.remove(child_de)
    tree.remove(child_c)
    print(tree)
    print(f"Re-Adding Children to Tree:")
    child_de = tree.addChild(child_dd, "fall wind")
    child_c = tree.addChild(child_a, "a child of spring")
    print(tree)

    # height
    print(f"Max Edges to the furthest Leaf Node: (Height) From:[{child_a.value}] {tree.height(child_a)}")

    # depth
    print(f"Edges from root to this node: (Depth) From[{child_c.value}]: {tree.depth(child_c)}")

    # parent node
    parent_of = tree.parent(child_c)
    print(f"Find the parent of the of the following Node:[{child_c.value}] Parent: {parent_of.value}")

    # children nodes
    child_of = tree.children(child_a)
    print(f"Find the children of the of the following Node:[{child_a.value}] Children: {[node.value for node in child_of]}")

    # test num children
    print(f"Testing num_children on node: [{child_d.value}] - Number of Children: {tree.num_children(child_a)}")

    # test iteration via different traversal algos
    print(f"\nDFS Search: (Always goes top -> bottom, then left -> right)\n{tree.preorder()}")
    print(f"\nReverse DFS Search: (Always goes bottom -> top, then left -> right)\n{tree.postorder()}")
    print(f"\nBFS Search: (go by levels - left to right, then down to the next level)\n{tree.level_order()}")

   
    print(f"\n{tree.flattened_view()}")
    print(f"Testing __Contains__: {'fall colors' in tree}")
    print(f"Testing __Contains__: {'432432' in tree}")

    print(tree.bfs_view())


if __name__ == "__main__":
    main()
