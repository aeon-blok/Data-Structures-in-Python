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
from utils.representations import AVLTreeRepr
from utils.helpers import RandomClass
from utils.exceptions import *

from adts.collection_adt import CollectionADT
from adts.bst_adt import BinarySearchTreeADT, iBSTNode

from ds.sequences.Stacks.array_stack import ArrayStack
from ds.primitives.arrays.dynamic_array import VectorArray, VectorView
from ds.trees.tree_nodes import BSTNode, AvlNode
from ds.trees.tree_utils import TreeUtils

# endregion


"""
AVL Tree: A self balancing Binary Search Tree.
After every insertion & deletion there is a rebalancing operation 
that involves different types of rotations.
The balance factor is the key property for an AVL tree -- every node has a balance factor -- -1, 0 or 1
whenever the balance factor for any node goes above or below this, the tree rebalances via rotations.
we select which rotation to use based on the balance factor.
"""


class AvlTree(BinarySearchTreeADT[T, K], CollectionADT[T], Generic[T, K]):
    """Avl Tree Implementation"""
    def __init__(self, datatype: type) -> None:
        self._root = None
        self._datatype = ValidDatatype(datatype)
        self._tree_keytype: None | type = None

        # Composed Objects:
        self._utils = TreeUtils(self)
        self._validators = DsValidation()
        self._desc = AVLTreeRepr(self)

    @property
    def keytype(self) -> Optional[type]:
        return self._tree_keytype

    @property
    def datatype(self) -> type:
        return self._datatype

    @property
    def root(self):
        return self._root

    @property
    def unbalanced_tree(self) -> bool:
        return self._utils.avl_tree_check_unbalanced(AvlNode)

    @property
    def max_balance_factor(self):
        return self._utils.avl_tree_max_balance_factor(AvlNode)

    # ----- Utility Operations -----
    def __str__(self) -> str:
        return self._desc.str_avl()

    def __repr__(self) -> str:
        return self._desc.repr_avl()

    # ----- Canonical ADT Operations -----
    def __len__(self) -> Index:
        return self._utils.binary_count_total_tree_nodes(AvlNode)

    def __contains__(self, key) -> bool:
        return self.search(key) is not None

    def __iter__(self):
        return self.inorder()

    def clear(self) -> None:
        self._utils.check_empty_binary_tree()
        self._root = None

    def is_empty(self) -> bool:
        return self._root is None

    # ----- Accessors -----
    def parent(self, node) -> iBSTNode[T, K] | None:
        self._utils.validate_tree_node(node, AvlNode)
        return node.parent

    def left(self, node) -> iBSTNode[T, K] | None:
        self._utils.validate_tree_node(node, AvlNode)
        return node.left if node.left else None

    def right(self, node) -> iBSTNode[T, K] | None:
        self._utils.validate_tree_node(node, AvlNode)
        return node.right if node.right else None

    def height(self) -> int:
        return self._utils.binary_tree_height()

    def search(self, key) -> iBSTNode[T, K] | None:
        self._utils.check_empty_binary_tree()
        key = Key(key)
        self._utils.check_key_is_same_type(key)
        return self._utils.bst_descent(self._root, AvlNode, key)

    def minimum(self, node) -> iBSTNode[T, K] | None:
        self._utils.check_empty_binary_tree()
        self._utils.validate_tree_node(node, AvlNode)
        while node.left is not None: node = node.left
        return node

    def maximum(self, node) -> iBSTNode[T, K] | None:
        self._utils.check_empty_binary_tree()
        self._utils.validate_tree_node(node, AvlNode)
        while node.right is not None: node = node.right
        return node

    def successor(self, node) -> iBSTNode[T, K] | None:
        """
        successor = next key greater than current key
        why do we do this? because the next larger key is not always directly connected to the current node.
        """
        # Case 1: Node has right child -- traverse down tree
        if node.right is not None:
            current_node = node.right  # go right 1 time.
            while current_node.left is not None:
                current_node = current_node.left
            return current_node  # return last node from left subtree

        # Case 2: Node has no right child -- climb up tree
        current_node = node
        parent_node = current_node.parent
        # climb tree whle current node is greater than parent.
        while parent_node is not None and current_node == parent_node.right:
            # traverse up tree
            current_node = parent_node
            parent_node = parent_node.parent

        return parent_node  # can be None if node is max key.

    def predecessor(self, node) -> iBSTNode[T, K] | None:
        """predecessor = next key less than current key"""
        # Case 1: Node has left child --traverse down tree
        if node.left is not None:
            current_node = node.left  # 1 time
            while current_node.right is not None:
                current_node = current_node.right
            return current_node  # last node

        # Case 2: Node has no left child -- climb up tree
        current_node = node
        parent_node = current_node.parent
        # this means -traverse up while the current node is less than the parent
        while parent_node is not None and current_node == parent_node.left:
            current_node = parent_node
            parent_node = parent_node.parent
        return parent_node  # can be none.

    # ----- Mutators -----
    def _avl_recursive_insert(self, node, key, value):
        """
        Recursive Insertion in an AVL tree is similar to BST trees. -- O(log N)
        """
        value = TypeSafeElement(value, self.datatype)
        key = Key(key)
        self._utils.check_key_is_same_type(key)

        # 1. If unoccupied - create and return new node.
        if node is None:
            return AvlNode(self.datatype, key, value, self)
        # 1.b if there is a match - replace the value and return the node
        elif key == node.key:
            node.element = value
            return node
        # 2. recursively travel until either finding a match, or the end of the tree
        if key < node.key:
            node.left = self._avl_recursive_insert(node.left, key, value)
        else:
            node.right = self._avl_recursive_insert(node.right, key, value)

        # 3. Update height
        node.update_height()

        # 5. Rebalance and rotate
        node = self._utils.rebalance_avl_tree(node)
        return node

    def insert(self, key, value) -> iBSTNode[T, K]:
        """public wrapper for inserting nodes into avl tree"""
        self._root = self._avl_recursive_insert(self._root, key, value)
        return self._root

    def _avl_recursive_delete(self, current_node, target_node):
        """
        This is a recursive helper method that deletes the target node from the subtree(subtree_root= current_node)
        Structural relinking is implicit in the return/assignment pattern.
        """

        # Base Case: must be the first line of code - we got to the end of the tree
        # the caller will assign none into its left or right child pointer.
        if current_node is None:
            return None

        # --- BST descent ---
        # If the target is smaller/larger, recursively traverse into the appropriate child and assign the result back to that child pointer.
        if target_node.key < current_node.key:
            current_node.left = self._avl_recursive_delete(current_node.left, target_node)
            # update parent pointer
            if current_node.left: current_node.left.parent = current_node
        elif target_node.key > current_node.key:
            current_node.right = self._avl_recursive_delete(current_node.right, target_node)
            # update parent pointer
            if current_node.right: current_node.right.parent = current_node
        else:
            # 1. Leaf Case: (0 children)
            # Deleting a leaf -> replace this spot with None. The caller assigned that None into its left/right and the node is removed.
            if current_node.left is None and current_node.right is None:
                return None

            # 2. 1 Child Case:
            # If exactly one child exists, return that child. The caller will assign the returned child into its left/right pointer,
            # effectively replacing the deleted node with its child (this is the relink)
            elif current_node.left is None:
                return current_node.right
            elif current_node.right is None:
                return current_node.left

            # 3. 2 children Case: (swap contents with succ - then delete succ.)
            else:
                succ = self.successor(current_node)
                current_node.key = succ.key
                current_node.element = succ.element
                current_node.right = self._avl_recursive_delete(current_node.right, succ)

        # 4. Update Height
        current_node.update_height()

        # 5. Rebalance tree.
        return self._utils.rebalance_avl_tree(current_node)

    def delete(self, node):
        """public wrapper for recursive deletion in AVL tree"""
        # Empty Case: raise error
        self._utils.check_empty_binary_tree()
        # validate node
        self._utils.validate_tree_node(node, AvlNode)
        old_value = node.element
        self._root = self._avl_recursive_delete(self._root, node)
        if self._root: self._root.parent = None # update parent pointer
        return old_value

    def replace(self, node, value):
        """replace the element value for a specific node"""
        old_value = node.element
        value = TypeSafeElement(value, self.datatype)
        node.element = value
        return old_value

    # ----- Traversals -----
    def preorder(self):
        return self._utils.binary_dfs_traversal(self._root, AvlNode)

    def postorder(self):
        return self._utils.binary_postorder_traversal(self._root, AvlNode)

    def levelorder(self):
        return self._utils.binary_bfs_traversal(self._root, AvlNode)

    def inorder(self):
        return self._utils.inorder_traversal(self._root, AvlNode)


# ---------------- Main -- Client Facing Code ----------------

def main():

    random_data = [
        "apple",
        "orange",
        "banana",
        "grape",
        "kiwi",
        "mango",
        "pear",
        "peach",
        "plum",
        "cherry",
        "lemon",
        "lime",
        "apricot",
        "blueberry",
        "strawberry",
        "raspberry",
        "blackberry",
        "papaya",
        "melon",
        "cantaloupe",
        "nectarine",
        "pomegranate",
        "fig",
        "date",
        "tangerine",
        "passionfruit",
        "guava",
        "lychee",
        "dragonfruit",
        "kumquat",
    ]

    random_data_subset = [
        "apple",
        "orange",
        "banana",
        "grape",
        "kiwi",
        "mango",
    ]

    avl = AvlTree(str)
    print(avl)
    print(repr(avl))

    print(f"\nTesting is empty?: {avl.is_empty()}")
    print(f"\nTesting Clear on Empty tree")
    try:
        avl.clear()
    except Exception as e:
        print(e)

    print(f"\nTesting access of root property on empty tree")
    try:
        print(avl.root)
    except Exception as e:
        print(e)

    random_keys = [i for i in range(100)]
    key_sample = random.sample(random_keys, 30)

    for keys, data in zip(key_sample, random_data):
        avl.insert(keys, data)

    keyhhs = [5, 3, 8, 3, 5, 7]
    for k in keyhhs:
        avl.insert(k, f"VALUE {k}")

    print(avl)
    print(repr(avl))

    print(f"\nTesting Replace operation:")
    print(f"root value = {avl.root.element}")
    old_value = avl.replace(avl.root, "DA NEW ROOT BAWSSSS")
    print(f"old replaced value = {old_value}")
    print(f"new root value = {avl.root.element}")
    print(avl)

    print(f"Inorder Traversal: {[i.element for i in avl.inorder()]}")

    print(f"\nTesting Deletion on a subset of items")
    keys_list = [i for i in avl.inorder()]
    keys_subset = keys_list[:10]
    print(f"items to delete: {len(keys_subset)}")
    print(f"Items: {', '.join([i.element for i in keys_subset])}")
    for i in keys_subset:
        avl.delete(i)
    print(avl)
    print(f"Is the item there?: {[i.element for i in avl.inorder()]}")

    max = avl.maximum(avl.root)
    parent_of_max = avl.parent(max)
    min = avl.minimum(avl.root)
    left_child_of_min = avl.left(min)
    pred = avl.predecessor(max)
    succ = avl.successor(min)
    right_child_of_root = avl.right(avl._root)

    print(f"\nFinding max key: {max}")
    print(f"Finding min key: {min}")
    print(f"Finding Successor to minimum key: {succ}")
    print(f"Finding Predecessor to max key: {pred}")
    print(f"Finding Parent of max key: {parent_of_max}")
    print(f"Finding left child of min key: (should be None!): {left_child_of_min}")
    print(f"Finding right child of root: {right_child_of_root}")


if __name__ == "__main__":
    main()
