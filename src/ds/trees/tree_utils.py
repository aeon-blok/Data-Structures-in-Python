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
from collections.abc import Sequence

# endregion


# region custom imports
from utils.custom_types import T
from utils.validation_utils import DsValidation
from utils.exceptions import *
from utils.helpers import Ansi

if TYPE_CHECKING:
    from adts.collection_adt import CollectionADT
    from adts.tree_adt import TreeADT, iTNode

from ds.primitives.arrays.dynamic_array import VectorArray
from ds.sequences.Stacks.array_stack import ArrayStack
from ds.sequences.Deques.circular_array_deque import CircularArrayDeque

# endregion

class TreeNodeUtils:
    """Utility Methods for Tree Nodes"""
    def __init__(self, tree_node_obj: "iTNode[T]") -> None:
        self.obj = tree_node_obj

    def validate_tnode(self, node):
        """ensures the specified child belongs to this node."""
        if node is None:
            raise NodeEmptyError("Error: Node is None.")
        from adts.tree_adt import iTNode
        if not isinstance(node, iTNode):
            raise DsTypeError("Error: Node is not a valid Node Type.")
        if node.deleted:
            raise NodeDeletedError(f"Error: Node has already been deleted.")
        if node not in self.obj.children:  # existence check
            raise NodeOwnershipError(f"Error: Node {node} is not a child of this node.")


class TreeUtils:
    """A collection of reusable utility methods for tree data structures"""
    def __init__(self, tree_obj: "TreeADT[T]") -> None:
        self.obj = tree_obj
        self._ansi = Ansi()

    def validate_tree_node(self, node):
        """ensures tree node is valid and belongs to the tree"""
        if node is None:
            raise NodeEmptyError("Error: Node is None.")
        from adts.tree_adt import iTNode
        if not isinstance(node, iTNode):
            raise DsTypeError("Error: Node is not a valid Node Type.")
        if node.deleted:
            raise NodeDeletedError("Error: This Node has been deleted and cannot be utilized in tree networks.")
        if node._tree_owner is not self.obj:
            raise NodeOwnershipError("Error: Node Belongs to a different Tree...")

    def count_total_tree_nodes(self) -> int:
        """Counts the total number of nodes in the tree. -- traverses whole tree - O(N)"""
        # empty case:
        if self.obj.root is None:
            return 0
        # main case: traverse tree - and count nodes
        from adts.tree_adt import iTNode
        tree_nodes = ArrayStack(iTNode)
        tree_nodes.push(self.obj.root)
        total_nodes = 0
        while tree_nodes:
            node = tree_nodes.pop()
            total_nodes += 1
            for i in node.children:
                tree_nodes.push(i)
        return total_nodes

    def dfs_depth_first_search(self):
        """
        Depth First Search: (DFS) -- travels from root to last child 
        First goes (top -> bottom) then (left -> right)
        """
        # empty case:
        if not self.obj.root:
            return 
        # main case
        from adts.tree_adt import iTNode
        tree_nodes = ArrayStack(iTNode)
        tree_nodes.push(self.obj.root)
        # traverse tree - add children in reverse order to the stack.
        while tree_nodes:
            node = tree_nodes.pop()
            yield node.value
            for i in reversed(node.children):
                tree_nodes.push(i)

    def reverse_dfs_postorder_search(self):
        """
        generator for postorder traversal 
        goes bottom to top, left to right.
        Uses 2 stack technique to reverse the order.
        """
        if not self.obj.root:
            return
        # main case
        from adts.tree_adt import iTNode
        tree_nodes = ArrayStack(iTNode)
        tree_nodes.push(self.obj.root)

        reverse_stack = ArrayStack(iTNode)

        while tree_nodes:
            node = tree_nodes.pop()
            reverse_stack.push(node)
            for i in node.children:
                tree_nodes.push(i)

        while reverse_stack:
            node = reverse_stack.pop()
            yield node.value

    def bfs_breadth_first_search(self):
        """
        generator for BFS - Breadth First Search
        goes level by level, left to right.
        Uses a deque for adding and removing from both ends -- O(1) time
        """
        if not self.obj.root:
            return
        from adts.tree_adt import iTNode
        tree_nodes = CircularArrayDeque(iTNode)
        tree_nodes.add_rear(self.obj.root)
        while tree_nodes:
            node = tree_nodes.remove_front()
            yield node.value
            for i in node.children:
                tree_nodes.add_rear(i)

    def tree_depth(self, node):
        """returns Number of edges from the ROOT to the specified node -- traverse up parents until root."""
        depth = 0  # tracks the level from target node
        current_node = node
        while current_node.parent:
            current_node = current_node.parent
            depth += 1
        return depth

    def tree_height(self, node):
        """returns Max Number of edges from a specified node to a leaf node (no children). -- Algorithm: recursively compute the max height of children."""
        if self.obj.is_leaf(node):  # leaf nodes have 0 height
            return 0
        max_height = max(self.obj.height(i) for i in node.children)
        return 1 + max_height   # height must be 1 or over, because not a leaf

    def view_bfs(self):
        """BFS Visualization - splits nodes by level."""
        if not self.obj.root:
            return f"\nTree: (Breadth First Search) 🌳: Total Nodes: {len(self.obj)}\n"

        from adts.tree_adt import iTNode
        # store tree in a deque
        tree_nodes = tree_nodes = CircularArrayDeque(iTNode)
        tree_nodes.add_rear(self.obj.root)
        current_level = 0
        infostring_stack = ArrayStack(str)

        # traverses tree
        while tree_nodes:
            current_level_size = len(tree_nodes)
            # temp stack to store just the nodes of the specified level.
            level_node_elements = ArrayStack(self.obj.datatype)
            # iterates through all the nodes in the current level.
            for _ in range(current_level_size):
                # pop the oldest item. (from the tree deque)
                node = tree_nodes.remove_front()
                # add to the level stack
                level_node_elements.push(node.value)
                # enqueue children (add to rear of the tree deque)
                for i in node.children:
                    tree_nodes.add_rear(i)
            # generate level string.
            bfs_level_string = f"Level: {current_level}: {', '.join(level_node_elements)}"
            infostring_stack.push(bfs_level_string)
            current_level += 1

        # Generate final string:
        title = self._ansi.color(f"Tree: (Breadth First Search) 🌳:", Ansi.BLUE)
        tree_height = self.obj.height(self.obj.root)
        return f"\n{title}\nTotal Nodes: {len(self.obj)}, Tree Height: {tree_height}\n" + "\n".join(infostring_stack)
