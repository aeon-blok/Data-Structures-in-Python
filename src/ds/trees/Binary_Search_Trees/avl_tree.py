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
    def keytype(self):
        return self._tree_keytype

    @property
    def datatype(self):
        return self._datatype

    @property
    def root(self):
        return self._root

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

    def __str__(self) -> str:
        return self._desc.str_avl()

    def __repr__(self) -> str:
        return self._desc.repr_avl()

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
    def insert(self, key, value) -> iBSTNode[T, K]:
        """
        Insertion in an AVL tree is similar to BST trees. -- O(log N)
        Step 1: Validate Inputs
        Step 2: Empty Tree Case: Insert new node at Root and return
        Step 3: Traverse Tree: We are searching for a key match.
            if a match is found, replace the element with our new input element value
            else recursively move left and right until we come to the end of the tree.
            if the key is smaller = Move Left
            if the key is larger = Move Right
        Step 4: Create a New Node for the end of the tree.
        Step 5: Attach the new node to its parent node. (which we stored earlier)
        Step 6: Rebalance AVL Tree if necessary
            Check the balance factor for the node.
            depending on the value - execute rotation of the tree nodes.
            there are 4 types of rotations
            Once the height balance property of the AVL tree is restored, the node will be returned. 
        """
        value = TypeSafeElement(value, self.datatype)
        key = Key(key)
        self._utils.check_key_is_same_type(key)

        # Empty Tree Case: new node is the root. (no rebalancing necessary)
        if self._root is None:
            self._root = AvlNode(self.datatype, key, value, self)
            return self._root

        # traverse tree
        parent_node = self._root
        current_node = self._root

        while current_node:
            # store previous node. (for attaching our new node to the tree later)
            parent_node = current_node
            # if the new key is smaller - move left.
            if key < current_node.key:
                current_node = current_node.left
            # if the new key is larger - move right.
            elif key > current_node.key:
                current_node = current_node.right
            # if the element already exists - replace with new element value and return node (no rebalance)
            elif key == current_node.key:
                current_node.element = value
                return current_node

        # Unoccupied Slot (None): at the end of the tree.
        new_node = AvlNode(self.datatype, key, value, self)

        # now we must attach it to the tree. -- we attach it to the parent node stored above.
        if key < parent_node.key:
            parent_node.left = new_node
        else:
            parent_node.right = new_node

        # rebalances the tree to maintain AVL height balance property.
        return self._utils.rebalance_avl_tree(new_node)

    def delete(self, node) -> T:
        """
        Empty Tree Case: raise error
        Step 2: Validate input
        Node has two children → replace the node with in-order successor (or predecessor), then delete that node
        Node has one child → replace the node with its child
        After Deletion - we need to rebalance and rotate the tree
        we need to reverse traverse from the deletion point - and rebalance each node.
        we update the height for each node to check if we need to rebalance.
        """

        # Empty Case: raise error

        # validate node
        self._utils.validate_tree_node(node, AvlNode)

        # initialize variables
        old_value = node.element
        parent = node.parent    # used to reconnect swapped nodes to the tree again

        # 2 Children Case
        if node.left and node.right:
            succ = node.right
            succ_parent = node
            while succ.left:
                succ_parent = succ
                succ = succ.left

            # swap successor with node (we logically swap, the nodes stay in place, but the contents are swapped)
            node.key = succ.key
            node.element = succ.element

            # delete successor (move focus to the successor node - which now contains our target nodes items for deletion)
            node = succ
            parent = succ_parent

        # 1 Child Case or leaf - 2 case has already ran, so we know the target node has 1 child at most.
        # we aim to delete node (target node) - swap with its child (either left or right)
        child = node.left if node.left else node.right

        # root case - if the node is the root can just dereference it. (works for leaf and 1 child)
        if parent is None:
            self._root = child
            # because its the new root - root has no parents so dereference parent
            if child: 
                child.parent = None
        else:
            # check if child is left or right child. swap with node
            # this removes the node from the tree, while keeping the Sorted BST property intact.
            if parent.left == node:
                parent.left = child
            else:
                parent.right = child

            # relink to tree - child must point to the parent.
            if child:
                child.parent = parent

        # iteratively rebalance nodes starting from parent (which is now the succ node)
        current_node = parent
        while current_node:
            current_node.update_height()
            current_node = self._utils.rebalance_avl_tree(current_node)
            current_node = current_node.parent
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
    print(avl)
    print(repr(avl))

    print(f"\nTesting Replace operation:")
    print(f"root value = {avl.root.element}")
    old_value = avl.replace(avl.root, "DA NEW ROOT BAWSSSS")
    print(f"old replaced value = {old_value}")
    print(f"new root value = {avl.root.element}")
    print(avl)

    print(f"\nTesting Deletion on a subset of items")
    keys_list = [i for i in avl.inorder()]
    keys_subset = keys_list[:1]
    print(f"items to delete: {len(keys_subset)}")
    print(f"{', '.join([i.element for i in keys_subset])}")
    for i in keys_subset:
        avl.delete(i)
    print(f"{[i.element for i in avl.inorder()]}")
    print(avl)

    # for i in avl:
    #     print(i)


if __name__ == "__main__":
    main()
