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
from user_defined_types.generic_types import T
from utils.validation_utils import DsValidation
from utils.representations import BinaryTreeRepr
from utils.helpers import RandomClass
from utils.exceptions import *

from adts.collection_adt import CollectionADT
from adts.binary_tree_adt import BinaryTreeADT, iBNode

from ds.sequences.Stacks.array_stack import ArrayStack
from ds.primitives.arrays.dynamic_array import VectorArray, VectorView
from ds.trees.tree_nodes import BinaryNode
from ds.trees.tree_utils import TreeUtils
# endregion


class BinaryTree(BinaryTreeADT[T], CollectionADT[T], Generic[T]):
    """
    Basic Binary Tree: using linked nodes for the backbone.
    """
    def __init__(self, datatype:type) -> None:
        self._datatype = datatype
        self._root = None

        # composed objects
        self._utils = TreeUtils(self)
        self._validators = DsValidation()
        self._desc = BinaryTreeRepr(self)
        self._validators.validate_datatype(self._datatype)

    @property
    def datatype(self):
        return self._datatype
    
    

    # ----- Meta Collection ADT Operations -----
    def is_empty(self) -> bool:
        return self._root is None

    def __contains__(self, value: T) -> bool:
        """checks if the value exists within the tree. uses DFS traversal"""
        self._validators.enforce_type(value, self._datatype)
        if not self._root:
            return False
        tree_nodes = ArrayStack(iBNode)
        tree_nodes.push(self._root)
        while tree_nodes:
            current_node = tree_nodes.pop()
            if current_node.element == value:
                return True
            if current_node.right is not None:
                tree_nodes.push(current_node.right)
            if current_node.left is not None:
                tree_nodes.push(current_node.left)
        return False
         
    def __len__(self) -> int:
        """counts the number of tree nodes"""
        return self._utils.binary_count_total_tree_nodes(iBNode)

    def __iter__(self):
        """iterates through the tree and returns a list of the results. Uses DFS"""
        return [i for i in self._utils.binary_dfs_traversal(self._root, iBNode)]

    def clear(self) -> None:
        self._utils.check_empty_binary_tree()
        # self.delete(self._root)
        self._root = None

       
    # ----- Utilities -----
    def __str__(self) -> str:
        return self._desc.str_binary_tree()
    
    def __repr__(self) -> str:
        return self._desc.repr_binary_tree()
    
    # ----- Canonical ADT Operations -----
    # ----- Accessors -----
    @property
    def root(self):
        """return root node."""
        if not self._root:
            return None
        return self._root

    def parent(self, node):
        """return parent node of specified node"""
        self._utils.check_empty_binary_tree() # empty tree case:
        self._utils.validate_tree_node(node, iBNode)
        # root edge case: - root parent is always none
        if node == self._root:
            return None
        
        return node.parent

    def left(self, node):
        """return left child node of specified node"""
        self._utils.check_empty_binary_tree() # empty tree case:
        self._utils.validate_tree_node(node, iBNode)
        # left node exists case:
        if not node.left:
            return None
        return node.left

    def right(self, node):
        """returns right child node of specified node"""
        self._utils.check_empty_binary_tree() # empty tree case:
        self._utils.validate_tree_node(node, iBNode)
        # right node exists case:
        if not node.right:
            return None
        return node.right

    def num_children(self, node):
        """number of children for the specified node"""
        self._utils.check_empty_binary_tree() # empty tree case:
        self._utils.validate_tree_node(node, iBNode)
        return node.num_children()
    
    def height(self):
        """returns the max number of edges from root to the furthest leaf"""
        return self._utils.binary_tree_height()

    def depth(self, node):
        """returns the number of edges from root to specified node."""
        self._utils.validate_tree_node(node, iBNode)
        return self._utils._tree_depth(node, iBNode)
    

    # ----- Mutators -----
    def add_root(self, element):
        """
        Adds a root node to the tree...
        validate the inputs, create a new node, check if tree is empty, raise error if not.
        """
        self._validators.enforce_type(element, self._datatype)
        new_node = BinaryNode(self._datatype, element, tree_owner=self)
        if self.is_empty():
            self._root = new_node
            return self._root
        else:
            raise NodeExistenceError("Error: Root Node & tree already exists.")

    def add_left(self, element, node):
        """adds a new left child node of the specified reference node"""
        # empty tree case:
        self._utils.check_empty_binary_tree()
        # validate inputs
        self._validators.enforce_type(element, self._datatype)
        self._utils.validate_tree_node(node, iBNode)

        # left node exists case:
        if node.left:
            raise NodeExistenceError("Error: Left Child already exists.")
        # main case:
        new_node = BinaryNode(self._datatype, element, tree_owner=self)
        # link to tree.
        node.left = new_node
        new_node.parent = node
        return new_node
       
    def add_right(self, element, node):
        """adds a new node as the right child of the specified node."""
        # empty tree case:
        self._utils.check_empty_binary_tree()
        # validate inputs
        self._validators.enforce_type(element, self._datatype)
        self._utils.validate_tree_node(node, iBNode)

        # right node exists case:
        if node.right:
            raise NodeExistenceError("Error: Right Child already exists.")

        # main case:
        # create new node
        new_node = BinaryNode(self._datatype, element, tree_owner=self)
        # link to tree.
        node.right = new_node
        new_node.parent = node
        return new_node
    
    def replace(self, element, node):
        """replaces the element value of the specified node."""
        # empty tree case:
        self._utils.check_empty_binary_tree()
        # validate inputs
        self._validators.enforce_type(element, self._datatype)
        self._utils.validate_tree_node(node, iBNode)
        old_value = node.element    # store old value
        node.element = element  # replace value
        return old_value

    def delete(self, node):
        """deletes the specified node and reorganizes the tree"""
        # empty tree case:
        self._utils.check_empty_binary_tree()
        # validate inputs        
        self._utils.validate_tree_node(node, iBNode)
        old_value = node.element    # store value

        if node is self._root:
            self._root = None

        # Step 1: unlink the parent of the node first.
        parent_node = node.parent
        if parent_node:
            if parent_node.left is node: parent_node.left = None
            else: parent_node.right = None
        
        # Step 2: now disconnect the node from the parent.
        node.parent = None
        node.deleted = True
        node.tree_owner = None
    
        # Step 3: traverse node subtree - delete every element.
        # Using a stack = Zero recursion depth errors (999 items)
        # two stack technique to guarantee correct deletion order.
        subtree_nodes = ArrayStack(iBNode)
        subtree_nodes.push(node)
        reverse_stack = ArrayStack(iBNode)

        while subtree_nodes:
            current_node = subtree_nodes.pop()
            reverse_stack.push(current_node)    # push onto reverse stack (these will be dereferenced)
            if current_node.left is not None:
                subtree_nodes.push(current_node.left)   # push to main stack
            if current_node.right is not None:
                subtree_nodes.push(current_node.right)

        # deleting node pointers via postorder (in reverse) -- ensures that children are processed first, and parents processed last.
        # no node is deleted while still referenced. -- Works even with degenerate trees (linked-list shape)
        for i in reversed(reverse_stack):
            i.left = None
            i.right = None
            i.parent = None
            i.deleted = True
            i.tree_owner = None
            

        return old_value

    # ----- Traversals -----
    def preorder(self):
        return self._utils.binary_dfs_traversal(self._root, iBNode)

    def postorder(self):
        return self._utils.binary_postorder_traversal(self._root, iBNode)

    def levelorder(self):
        return self._utils.binary_bfs_traversal(self._root, iBNode)

    def inorder(self):
        return self._utils.inorder_traversal(self._root, iBNode)


# Main ---- Client Facing Code ----

def main():
    bt = BinaryTree(str)
    print("\ntesting Empty Tree")
    print(repr(bt))
    print(bt)
    print(f"Is the binary tree empty?: {bt.is_empty()}")
    print(f"is a random value in the tree?: {'2543' in bt}")

    try:
        bt.delete(bt.root)
    except Exception as e:
        print(e)

    try:
        bt.clear()
    except Exception as e:
        print(e)

    print(f"\nAdding items to Binary Tree")
    root = bt.add_root("Equidae")
    level_1a = bt.add_left("donkeys", root)
    level_1b =bt.add_right("horses", root)
    level_2a = bt.add_left("zebras", level_1a)
    level_2b = bt.add_right("Wild asian ass..", level_1a)
    level_2c = bt.add_left("NOT A HOrSE", level_1b)
    level_2d = bt.add_right("Kiang", level_1b)
    level_3a = bt.add_left("mule", level_2a)
    level_3b = bt.add_right("hinny", level_2a)
    level_4a = bt.add_left("new onager", level_3a)
    level_4b = bt.add_right("lorse", level_3a)
    print(bt)

    print(f"\nTesting Deletion of items for binary tree")
    delete_random = bt.delete(level_3a)
    print(delete_random)
    print(bt)
    try:
        test_deleted = bt.delete(level_4a)
        print(test_deleted)
    except Exception as e:
        print(e)

    print(f"\nTesting replace functionality")
    old_element = bt.replace("Elephant", level_2d)
    print(old_element)
    print(bt)
    print(repr(bt))

    print(f"\nTesting Type Safety")
    try:
        bt.add_left(RandomClass("GFDGFDGDF"), level_2d)
    except Exception as e:
        print(e)

    print(f"testing node type safety.")
    try:
        bt.add_left("GFDGFDGDF", "level_2d")
    except Exception as e:
        print(e)

    print(f"Testing DFS preorder traversal")
    preorder = [i for i in bt.preorder()]
    print(preorder)

    print(f"Testing Postorder Traversal")
    postorder = [i for i in bt.postorder()]
    print(postorder)

    print(f"Testing BFS levelorder traversal")
    levelorder = [i for i in bt.levelorder()]
    print(levelorder)

    print(f"testing inorder")
    inorder = [i for i in bt.inorder()]
    print(inorder)
    parent_hinny = bt.parent(level_3b)
    # print(parent_hinny)
    left_zebras = bt.left(level_2a)
    right_zebras = bt.right(level_2a)
    print(left_zebras)
    print(right_zebras)

    print(f"\nTesting Node Depth - from root, and specified node.")
    print(bt.depth(bt.root))
    print(bt.depth(level_3b))
    print(bt.height())

    print(f"\nTesting Num Children.")
    print(bt.num_children(level_3b))
    print(bt.num_children(bt.root))

    print(f"Testing Clear()")
    print(bt)
    bt.clear()
    print(bt)


if __name__ == "__main__":
    main()
