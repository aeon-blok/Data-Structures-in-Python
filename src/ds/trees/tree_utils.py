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
from utils.custom_types import T, K, Key
from utils.validation_utils import DsValidation
from utils.exceptions import *
from utils.helpers import Ansi

if TYPE_CHECKING:
    from adts.collection_adt import CollectionADT
    from adts.tree_adt import TreeADT, iTNode

from ds.primitives.arrays.dynamic_array import VectorArray
from ds.sequences.Stacks.array_stack import ArrayStack
from ds.sequences.Deques.circular_array_deque import CircularArrayDeque
from adts.tree_adt import iTNode
from adts.binary_tree_adt import iBNode

# endregion

class TreeNodeUtils:
    """Utility Methods for Tree Nodes"""
    def __init__(self, tree_node_obj) -> None:
        self.obj = tree_node_obj

    def validate_node(self, node, node_type: type):
        """ensures the specified child belongs to this node."""
        if node is None:
            raise NodeEmptyError("Error: Node is None.")
        if not isinstance(node, node_type):
            raise DsTypeError("Error: Node is not a valid Node Type.")
        if node.deleted:
            raise NodeDeletedError(f"Error: Node has already been deleted.")
        if node not in self.obj.children:  # existence check
            raise NodeOwnershipError(f"Error: Node {node} is not a child of this node.")

    def validate_node_binary_search_key(self, key):
            """ensures the the input key, is a valid key."""
            if not isinstance(key, Key):
                raise KeyInvalidError("Error: Input Key is not valid. All keys must be hashable, immutable & comparable (<, >, ==, !=)")
            elif key is None:
                raise KeyInvalidError("Error: Key cannot be None Value")

    
class TreeUtils:
    """A collection of reusable utility methods for tree data structures"""
    def __init__(self, tree_obj) -> None:
        self.obj = tree_obj
        self._ansi = Ansi()

    # region general tree
    def validate_tree_node(self, node, node_type: type):
        """ensures tree node is valid and belongs to the tree"""
        self.validate_datatype(node_type)
        if node is None:
            raise NodeEmptyError("Error: Node is None.")
        elif not isinstance(node, node_type):
            raise DsTypeError("Error: Node is not a valid Node Type.")
        elif node.deleted:
            raise NodeDeletedError("Error: This Node has been deleted and cannot be utilized in tree networks.")
        if node.tree_owner is not self.obj:
            raise NodeOwnershipError("Error: Node Belongs to a different Tree...")

    def validate_datatype(self, datatype):
        if datatype is None:
            raise DsUnderflowError("Error: Datatype cannot be None Value.")
        if not isinstance(datatype, type):
            raise DsTypeError("Error: Datatype must be a valid Python Type object.")

    def count_total_tree_nodes(self, node_type: type) -> int:
        """Counts the total number of nodes in the tree. -- traverses whole tree - O(N)"""
        self.validate_datatype(node_type)
        # empty case:
        if self.obj.root is None:
            return 0
        # main case: traverse tree - and count nodes
        tree_nodes = ArrayStack(node_type)
        tree_nodes.push(self.obj.root)
        total_nodes = 0
        while tree_nodes:
            node = tree_nodes.pop()
            total_nodes += 1
            for i in node.children:
                tree_nodes.push(i)
        return total_nodes

    def _dfs_depth_first_search(self, target_node, node_type: type):
        """
        Depth First Search: (DFS) -- travels from root to last child 
        First goes (top -> bottom) then (left -> right)
        """
        self.validate_datatype(node_type)

        self.validate_tree_node(target_node, node_type)  # validate input

        # empty case:
        if not target_node:
            return

        # main case
        tree_nodes = ArrayStack(node_type)
        tree_nodes.push(target_node)

        # traverse tree - add children in reverse order to the stack.
        while tree_nodes:
            node = tree_nodes.pop()
            yield node.element
            for i in reversed(node.children):
                tree_nodes.push(i)

    def _reverse_dfs_postorder_search(self, target_node, node_type: type):
        """
        generator for postorder traversal 
        goes bottom to top, left to right.
        Uses 2 stack technique to reverse the order.
        """
        self.validate_datatype(node_type)
        self.validate_tree_node(target_node, node_type)

        if not target_node:
            return

        # main case
        tree_nodes = ArrayStack(node_type)
        tree_nodes.push(target_node)
        reverse_stack = ArrayStack(node_type)

        while tree_nodes:
            node = tree_nodes.pop()
            reverse_stack.push(node)
            for i in node.children:
                tree_nodes.push(i)

        while reverse_stack:
            node = reverse_stack.pop()
            yield node.element

    def _bfs_breadth_first_search(self, target_node, node_type: type):
        """
        generator for BFS - Breadth First Search
        goes level by level, left to right.
        Uses a deque for adding and removing from both ends -- O(1) time
        """

        self.validate_datatype(node_type)
        self.validate_tree_node(target_node, node_type)

        if not self.obj.root:
            return
        tree_nodes = CircularArrayDeque(iTNode)
        tree_nodes.add_rear(self.obj.root)
        while tree_nodes:
            node = tree_nodes.remove_front()
            yield node.element
            for i in node.children:
                tree_nodes.add_rear(i)

    def _tree_depth(self, node, node_type):
        """returns Number of edges from the ROOT to the specified node -- traverse up parents until root."""
        self.validate_datatype(node_type)
        self.validate_tree_node(node, node_type)

        depth = 0  # tracks the level from target node
        current_node = node
        while current_node.parent:
            current_node = current_node.parent
            depth += 1
        return depth

    def _tree_height(self, node):
        """returns Max Number of edges from a specified node to a leaf node (no children). -- Algorithm: recursively compute the max height of children."""
        if self.obj.is_leaf(node):  # leaf nodes have 0 height
            return 0
        max_height = max(self._tree_height(i) for i in node.children)
        return 1 + max_height   # height must be 1 or over, because not a leaf

    def view_bfs(self, node_type):
        """BFS Visualization - splits nodes by level."""
        if not self.obj.root:
            return f"\nTree: (Breadth First Search) 🌳: Total Nodes: {len(self.obj)}\n"

        self.validate_datatype(node_type)

        # store tree in a deque
        tree_nodes = tree_nodes = CircularArrayDeque(node_type)
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
                level_node_elements.push(node.element)
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

    # endregion

    # region binary tree
    def check_empty_binary_tree(self):
        """ensures the tree is not empty when inserting left or right children."""
        if self.obj.is_empty():
            raise DsUnderflowError("Error: Tree is empty... Action was not performed")

    def binary_count_total_tree_nodes(self, node_type: type):
        """binary tree variant for counting the nodes in a tree"""
        self.validate_datatype(node_type)

        # empty case:
        if self.obj.root is None:
            return 0

        # main case: traverse tree - and count nodes
        tree_nodes = ArrayStack(node_type)
        tree_nodes.push(self.obj.root)
        total_nodes = 0
        while tree_nodes:
            current_node = tree_nodes.pop()
            total_nodes += 1
            # add children to the stack
            if current_node.right is not None:
                tree_nodes.push(current_node.right)
            if current_node.left is not None:
                tree_nodes.push(current_node.left)
        return total_nodes

    def binary_tree_height(self, edge_based: bool = True):
        """returns max height for binary tree..."""
        if self.obj.root is None:
            return 0
        start_depth = 0 if edge_based else 1
        tree_nodes = ArrayStack(tuple)  # note the type is a tuple.
        tree_nodes.push((self.obj.root, start_depth))
        max_height_counter = 0

        while tree_nodes:
            current_node, depth = tree_nodes.pop()
            max_height_counter = max(max_height_counter, depth)

            # add children to the stack
            if current_node.right is not None:
                tree_nodes.push((current_node.right, depth + 1))
            if current_node.left is not None:
                tree_nodes.push((current_node.left, depth + 1))

        return max_height_counter

    def binary_dfs_traversal(self, target_node, node_type:type):
        """depth first search for binary trees"""
        self.validate_datatype(node_type)
        self.validate_tree_node(target_node, node_type)  # validate input

        # empty case:
        if not target_node:
            return

        # main case
        tree_nodes = ArrayStack(node_type)
        tree_nodes.push(target_node)

        # traverse tree - add children in reverse order to the stack.
        while tree_nodes:
            current_node = tree_nodes.pop()
            yield current_node
            # NOTICE THE ORDER - its right to left - when pushing to the stack with dfs
            if current_node.right is not None:
                tree_nodes.push(current_node.right)
            if current_node.left is not None:
                tree_nodes.push(current_node.left)  # push to main stack

    def binary_postorder_traversal(self, target_node, node_type:type):
        """reversed dfs for binary trees"""
        self.validate_datatype(node_type)
        self.validate_tree_node(target_node, node_type)

        if not target_node:
            return

        # main case
        tree_nodes = ArrayStack(node_type)
        tree_nodes.push(target_node)
        reverse_stack = ArrayStack(node_type)

        while tree_nodes:
            current_node = tree_nodes.pop()
            reverse_stack.push(current_node)
            # NOTICE: the order is reversed for postorder.
            if current_node.left is not None:
                tree_nodes.push(current_node.left)
            if current_node.right is not None:
                tree_nodes.push(current_node.right)

        while reverse_stack:
            node = reverse_stack.pop()
            yield node

    def binary_bfs_traversal(self, target_node, node_type: type):
        """breadth first search for binary trees"""
        self.validate_datatype(node_type)
        self.validate_tree_node(target_node, node_type)

        if not target_node:
            return

        tree_nodes = CircularArrayDeque(node_type)
        tree_nodes.add_rear(target_node)

        while tree_nodes:
            current_node = tree_nodes.remove_front()
            yield current_node
            if current_node.left is not None:
                tree_nodes.add_rear(current_node.left) 
            if current_node.right is not None:
                tree_nodes.add_rear(current_node.right)

    def inorder_traversal(self, target_node, node_type: type):
        """
        Inorder traversal for binary trees. Visit nodes in the order Left → Root → Right.
        For a general binary tree, the values won’t be sorted (only for BST)
        """
        self.validate_datatype(node_type)
        # self.validate_tree_node(target_node, node_type)

        if target_node is None:
            return

        tree_nodes = ArrayStack(node_type)
        current_node = target_node

        while tree_nodes or current_node:
            while current_node:
                tree_nodes.push(current_node)
                # move along left subtree.
                current_node = current_node.left
            # once we get to the end of the subtree
            current_node = tree_nodes.pop()
            # return value
            yield current_node
            # move to the right subtree.
            current_node = current_node.right

    # endregion

    # region BST
    def validate_binary_search_key(self, key):
            """ensures the the input key, is a valid key."""
            if not isinstance(key, Key):
                raise KeyInvalidError("Error: Input Key is not valid. All keys must be hashable, immutable & comparable (<, >, ==, !=)")
            elif key is None:
                raise KeyInvalidError("Error: Key cannot be None Value")

    def bst_descent(self, node, node_type, key):
        """
        descent algorithm - traverses the bst
        node - key matches? return match
        key < node key? traverse left (left keys are smaller than the parent.)
        key > node key? traverse right (right keys are lareger than the parent.)
        in both these cases traversal will continue until no more nodes are found (return None)
        All authoritative BST definitions treat search as a presence test, not “give me the last node you touched”.
        """
        self.validate_datatype(node_type)
        self.validate_binary_search_key(key)
        self.validate_tree_node(node, node_type)
        while node is not None:
            # match found case:
            if key == node.key: return node
            # key < node key case:
            elif key < node.key: node = self.obj.left(node)
            # key > node key case:
            else: node = self.obj.right(node)
        return None
    
    def bst_parent_descent(self, node, node_type, key):
        """Descent algorithm that returns the last node and an existence flag instead of None."""
        self.validate_binary_search_key(key)
        self.validate_tree_node(node, node_type)

        last_node = None
        current_node = node
        match_exists = False

        while current_node is not None:
            last_node = current_node
            # match found case:
            if key == current_node.key: 
                match_exists = True
                return current_node, match_exists
            # key < node key case:
            elif key < current_node.key: 
                current_node = self.obj.left(current_node)
            # key > node key case:
            else: 
                current_node = self.obj.right(current_node)
        return last_node, match_exists

    # endregion
