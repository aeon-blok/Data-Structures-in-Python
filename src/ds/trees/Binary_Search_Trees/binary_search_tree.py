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
from user_defined_types.custom_types import T, K
from utils.validation_utils import DsValidation
from utils.representations import BSTRepr
from utils.helpers import RandomClass
from utils.exceptions import *

from adts.collection_adt import CollectionADT
from adts.bst_adt import BinarySearchTreeADT, iBSTNode

from ds.sequences.Stacks.array_stack import ArrayStack
from ds.primitives.arrays.dynamic_array import VectorArray, VectorView
from ds.trees.tree_nodes import BSTNode
from ds.trees.tree_utils import TreeUtils

# endregion


"""
Binary Search Tree: 
On every node p, all keys in the left subtree < p.key and all keys in the right subtree > p.key.
No duplicates unless you explicitly add a duplicate-policy.
"""

class BinarySearchTree(BinarySearchTreeADT[T, K], CollectionADT[T], Generic[T, K]):
    """Binary Search tree -- all keys are ordered with BST property."""
    def __init__(self, datatype) -> None:
        self._root = None
        self._datatype = datatype

        # composed objects
        self._utils = TreeUtils(self)
        self._validators = DsValidation()
        self._desc = BSTRepr(self)

    @property
    def datatype(self):
        return self._datatype

    @property
    def sibling(self):
        pass

    @property
    def root(self):
        if self._root is None:
            return None
        else:
            return self._root

    # ----- Meta Collection ADT Operations -----
    def is_empty(self) -> bool:
        return self._root is None

    def clear(self) -> None:
        self._utils.check_empty_binary_tree()
        self._root = None

    def __len__(self) -> int:
        return self._utils.binary_count_total_tree_nodes(iBSTNode)

    def __contains__(self, key) -> bool:
        self._utils.validate_binary_search_key(key)
        return self.search(key) is not None

    def __iter__(self):
        """Generate an iteration of all keys in the map in order."""
        return [i for i in self._utils.inorder_traversal(self._root, iBSTNode)]

    # ----- Utilities -----
    def __str__(self) -> str:
        return self._desc.str_bst()

    def __repr__(self) -> str:
        return self._desc.repr_bst()

    def __getitem__(self, key: K):
        pass

    def __setitem__(self, key: K, value: T):
        pass

    def __delitem__(self, node: iBSTNode[T, K]):
        pass

    # ----- Canonical ADT Operations -----
    # ----- Accessors -----
    def parent(self, node):
        self._utils.validate_tree_node(node, iBSTNode)
        return node.parent

    def left(self, node):
        self._utils.validate_tree_node(node, iBSTNode)
        return node.left

    def right(self, node):
        self._utils.validate_tree_node(node, iBSTNode)
        return node.right

    def height(self):
        """find the number of edges from the root to the furthest leaf"""
        return self._utils.binary_tree_height()

    def minimum(self, node):
        """
        find the minimum key in the specified node subtree
        Minimum: the leftmost node in a subtree (keep following left pointers until None).
        """
        self._utils.check_empty_binary_tree()
        self._utils.validate_tree_node(node, iBSTNode)
        while node.left is not None: node = node.left
        return node

    def maximum(self, node):
        """
        find the maximum key in the specified node subtree
        Maximum: the rightmost node in a subtree (keep following right pointers until None).
        """
        self._utils.check_empty_binary_tree()
        self._utils.validate_tree_node(node, iBSTNode)
        while node.right is not None: node = node.right
        return node

    def successor(self, node):
        """
        successor = next key greater than current key
        why do we do this? because the next larger key is not always directly connected to the current node.
        """
        # Case 1: Node has right child -- traverse down tree
        if node.right is not None:
            current_node = node.right # go right 1 time.
            while current_node.left is not None:
                current_node = current_node.left
            return current_node # return last node from left subtree

        # Case 2: Node has no right child -- climb up tree
        current_node = node
        parent_node = current_node.parent
        # climb tree whle current node is greater than parent.
        while parent_node is not None and current_node == parent_node.right:
            # traverse up tree
            current_node = parent_node
            parent_node = parent_node.parent

        return parent_node  # can be None if node is max key.

    def predecessor(self, node):
        """predecessor = next key less than current key"""
        # Case 1: Node has left child --traverse down tree
        if node.left is not None:
            current_node = node.left # 1 time
            while current_node.right is not None:
                current_node = current_node.right
            return current_node # last node

        # Case 2: Node has no left child -- climb up tree
        current_node = node
        parent_node = current_node.parent
        # this means -traverse up while the current node is less than the parent
        while parent_node is not None and current_node == parent_node.left:
            current_node = parent_node
            parent_node = parent_node.parent
        return parent_node  # can be none.

    def search(self, key: K):
        """searches for a node that matches a key. -- returns None if key not found -- O(H)"""
        self._utils.check_empty_binary_tree()
        self._utils.validate_binary_search_key(key)
        # returns none if key not found
        return self._utils.bst_descent(self._root, iBSTNode, key)

    def search_by_key(self, key):
        pass

    # ----- Mutators -----
    def insert(self, key, value):
        """Inserts a new node into the binary search tree. - O(H)"""
        self._validators.enforce_type(value, self._datatype)
        self._utils.validate_binary_search_key(key)
        new_node = BSTNode(self._datatype, key, value, tree_owner=self)
        # empty tree case:
        if self._root is None:
            self._root = new_node
            return self._root
        node, match_exists = self._utils.bst_parent_descent(self._root, iBSTNode, key)
        # match case: replace element with new element value
        if match_exists:
            node.element = value
            return node
        # match not found case: - insert a new node as the child of last node found.
        if key < node.key:
            node.left = new_node
            new_node.parent = node
            return node.left
        else:
            node.right = new_node
            new_node.parent = node
            return node.right

    def replace(self, node, value):
        """updates element value if found."""
        self._validators.enforce_type(value, self._datatype)
        self._utils.validate_tree_node(node, iBSTNode)
        old_value = node.element
        node.element = value
        return old_value

    def replace_by_key(self, key, value):
        pass

    def delete(self, node):
        """deletes a node from the binary search tree and reorganizes the tree."""
        self._utils.check_empty_binary_tree()
        self._utils.validate_tree_node(node, iBSTNode)
        old_value = node.element    # store old value

        # 2 child case:
        # find successor((smallest node in right subtree)) or predecessor (largest in left subtree)
        # swap elements (Copy successor’s key/value into the node to delete),
        # delete successor (now leaf or 1-child - same ops as above).
        if node.left and node.right:
            # find successor (smallest node in right subtree - can only have 1 child)
            successor = self.successor(node)
            # swap node element with successor element
            self.replace(node, successor.element)
            # Node is now swapped with sucessor node. (now only has 1 child.)
            node = successor

        # 1 child case / leaf case: -- relink the parent directly to the child (unlinks node)
        if node.left:
            child = node.left
        elif node.right:
            child = node.right
        else:   # leaf node (has 0 children)
            child = None

        # unlink node - connect child to node parent.
        if child is not None:
            child.parent = node.parent

        # update / relink parent pointers
        # root case: parent is the root. -- set root to child manually.
        if node.parent is None:
            self._root = child
        elif node == node.parent.left:
            node.parent.left = child
        else:
            node.parent.right = child

        # dereference
        node.parent = node.left = node.right = None
        node.deleted = None
        node.tree_owner = None

        return old_value

    def delete_by_key(self, key):
        pass       

    # ----- Traversals -----

    def preorder(self):
        return self._utils.binary_dfs_traversal(self._root, iBSTNode)

    def postorder(self):
        return self._utils.binary_postorder_traversal(self._root, iBSTNode)

    def levelorder(self):
        return self._utils.binary_bfs_traversal(self._root, iBSTNode)

    def inorder(self):
        return self._utils.inorder_traversal(self._root, iBSTNode)


# Main ----------- Client Facing Code ------------

# todo test upsert
# todo test 2 child delete, 1 child delete, delete root
# todo test search
# todo test basic accessors
# todo build __setitem__ functionality.
# todo add search, replace, delete by key.

def main():
    bst = BinarySearchTree(str)
    print(bst)
    print(repr(bst))
    print(f"Is tree empty?: {bst.is_empty()}")

    random_data = ['apple', 'orange', 'banana', 'grape', 'kiwi', 'mango', 'pear', 'peach', 'plum', 'cherry',
 'lemon', 'lime', 'apricot', 'blueberry', 'strawberry', 'raspberry', 'blackberry', 'papaya', 
 'melon', 'cantaloupe', 'nectarine', 'pomegranate', 'fig', 'date', 'tangerine', 'passionfruit', 
 'guava', 'lychee', 'dragonfruit', 'kumquat']

    random_keys = [i for i in range(100)]
    key_sample = random.sample(random_keys, 30)
    packed = (key_sample, random_data)

    for keys, data in zip(key_sample, random_data):
        bst.insert(keys, data)

    print(bst)
    inorder = [(i.key,i.element) for i in bst.inorder()]
    dfs = [(i.key,i.element) for i in bst.preorder()]
    postorder = [(i.key, i.element) for i in bst.postorder()]
    levelorder = [(i.key, i.element) for i in bst.levelorder()]
    print(f"\nInorder Traversal")
    print(inorder)
    print(f"\nPreorder Traversal")
    print(dfs)
    print(f"\nPostorder")
    print(postorder)
    print(f"\nlevelorder")
    print(levelorder)

    print(f"\nBST Min: {bst.minimum(bst.root)} and successor: {bst.successor(bst.minimum(bst.root))}")
    print(f"BST Maximum: {bst.maximum(bst.root)} and predecessor: {bst.predecessor(bst.maximum(bst.root))}")

    print(f"\ntesting __contains__: {324325 in bst}")
    max = bst.maximum(bst.root)
    max_del = bst.delete(max)
    print(max_del)
    print(len(bst))

    min = bst.minimum(bst.root)
    print(bst.successor(min))
    bst.replace(min, "NOT Tangerine")
    print(min)


if __name__ == "__main__":
    main()
