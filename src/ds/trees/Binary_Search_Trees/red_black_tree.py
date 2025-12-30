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
from utils.validation_utils import DsValidation
from utils.representations import RedBlackTreeRepr
from utils.helpers import RandomClass
from utils.exceptions import *

from adts.collection_adt import CollectionADT
from adts.bst_adt import BinarySearchTreeADT, iBSTNode

from ds.sequences.Stacks.array_stack import ArrayStack
from ds.primitives.arrays.dynamic_array import VectorArray, VectorView
from ds.trees.tree_nodes import BSTNode, AvlNode, RedBlackSentinel, RedBlackNode
from ds.trees.tree_utils import TreeUtils

from user_defined_types.key_types import iKey, Key
from user_defined_types.tree_types import NodeColor, Traversal

# endregion


"""
Red - Black Tree: 
Utilizes Color to constrain the height of the tree.
Properties:
Every Node Must be red or black
red property: a red node cannot have a red child
If a node has 1 child - it must be red
black property: every path from a node to its leaf nodes must go through the same number of nodes
black height - is total height / 2
all leaves and null nodes are black
black depth - number of black nodes from the root to the specified node. (number of black ancestors)
The root node is colored black.
"""


class RedBlackTree(BinarySearchTreeADT[T, K], CollectionADT[T], Generic[T, K]):
    """
    Implementation of a Red Black Tree
    This implementation uses sentinels - self.NIL - the sentinel is colored Black.
    You should NEVER return None inside RB-tree algorithms.
    """
    def __init__(self, datatype: type) -> None:
        self._datatype = datatype
        self._tree_keytype: None | type = None
        self.NIL = RedBlackSentinel(self.datatype, tree_owner=self)
        self.NIL.left = self.NIL.right = self.NIL.parent = self.NIL
        self._root = self.NIL

        # composed objects
        self._utils = TreeUtils(self)
        self._validators = DsValidation()
        self._desc = RedBlackTreeRepr(self)

    @property
    def root(self) -> iBSTNode[T, K] | RedBlackSentinel: # type: ignore
        return self._root

    @root.setter
    def root(self, value) -> None:
        self._root = value

    @property
    def datatype(self) -> type:
        return self._datatype

    @property
    def tree_keytype(self) -> Optional[type]:
        return self._tree_keytype

    @property
    def is_red_property(self) -> bool:
        """returns True if there are no red red violations in the tree - O(N) - iteratively traverses the entire tree"""
        # * root case
        if self._root == self.NIL:
            return True
        
        # init stack
        tree = ArrayStack(RedBlackNode, 5000)
        tree.push(self._root)

        # * traverse tree and check for red red violations.
        while tree:
            node = tree.pop()
            if node == self.NIL:
                continue

            left = node.left
            right = node.right
                
            if node.color == NodeColor.RED and (left.color == NodeColor.RED or right.color == NodeColor.RED):
                return False
            # * push children onto stack to iteratively traverse.
            if left != self.NIL:
                tree.push(left)
            if right != self.NIL:
                tree.push(right)
        return True
    
    @property
    def is_red_property_recursive(self) -> bool:
        def _check_color(node):
            if node == self.NIL:
                return True
            if node.color == NodeColor.RED:
                if node.left.color == NodeColor.RED or node.right.color == NodeColor.RED:
                    return False
            return _check_color(node.left) and _check_color(node.right)
        return _check_color(self.root)
    
    @property
    def is_black_property(self) -> bool:
        """Black Property Boolean. returns false if black properties are violated."""
        # invariants
        if self._root == self.NIL:
            return True
        if self._root.color != NodeColor.BLACK:
            return False
        
        # initialize tree for traversal: stores a tuple (node, black_count)
        tree = ArrayStack(tuple, 5000)
        tree.push((self._root, 0))
        start_path_black_count = None   # used to compare all future tree path iterations.

        while tree:
            node, black_count = tree.pop()
            # * End of Tree Path Case:
            if node == self.NIL:
                final_black_count = black_count + 1
                if start_path_black_count is None:
                    start_path_black_count = final_black_count
                # * exit condition - paths are not equal number of black nodes.
                elif start_path_black_count != final_black_count:
                    return False
                continue
            # increment count if node is black
            if node.color == NodeColor.BLACK:
                black_count += 1
            # add children to the tree for traversal (if they are not sentinels)
            if node.left != self.NIL:
                tree.push((node.left, black_count))
            if node.right != self.NIL:
                tree.push((node.right, black_count))
        # * exit condition - paths have equal number of black nodes.
        return True
                
    @property
    def is_black_property_recursive(self) -> bool:
        """
        ensures the black property is maintained.
        """
        # * root must always be black
        if self._root.color != NodeColor.BLACK:
            return False
        # if root is sentinel (its black)
        if self._root == self.NIL:
            return True

        def _inspect(node):
            # base case if leaf - its black by default.
            if node == self.NIL: 
                return 1
            # post order traverse
            left = _inspect(node.left)
            right = _inspect(node.right)
            # validate case: if no black nodes found or if the left and right subtrees arent equal - signal violation
            if left != right:
                # print(f"Violation at node {node.element}: left {left} != right {right}")
                return 0 # violation
            return left + (1 if node.color == NodeColor.BLACK else 0)

        return _inspect(self.root) != 0 

    # ----- Meta Collection ADT Operations -----
    def __len__(self) -> Index:
        return self._utils.red_black_count_tree_nodes(RedBlackNode)

    def __contains__(self, key) -> bool:
        return False if self.is_empty() else self.search(key) is not None

    def clear(self) -> None:
        self._root = self.NIL

    def is_empty(self) -> bool:
        return self._root == self.NIL

    def __iter__(self):
        return self.inorder()

    # ----- Utilities -----
    def __str__(self) -> str:
        return self._desc.str_redblack_tree()

    def __repr__(self) -> str:
        return self._desc.repr_redblack_tree()

    # ----- Canonical ADT Operations -----
    # ----- Accessors -----
    def parent(self, node: iBSTNode[T, K]) -> iBSTNode[T, K] | None:
        self._utils.red_black_is_sentinel(node)
        self._utils.validate_tree_node(node, RedBlackNode)
        return node.parent if not self.NIL else self.NIL  # type: ignore

    def left(self, node: iBSTNode[T, K]) -> iBSTNode[T, K] | None:
        self._utils.red_black_is_sentinel(node)
        self._utils.validate_tree_node(node, RedBlackNode)
        return node.left if not self.NIL else self.NIL  # type: ignore

    def right(self, node: iBSTNode[T, K]) -> iBSTNode[T, K] | None:
        self._utils.red_black_is_sentinel(node)
        self._utils.validate_tree_node(node, RedBlackNode)
        return node.right if not self.NIL else self.NIL  # type: ignore

    def minimum(self, node: iBSTNode[T, K]) -> iBSTNode[T, K] | None:
        self._utils.check_empty_binary_tree()
        self._utils.red_black_is_sentinel(node)
        self._utils.validate_tree_node(node, RedBlackNode)
        while node.left != self.NIL:
            node = node.left
        return node

    def maximum(self, node: iBSTNode[T, K]) -> iBSTNode[T, K] | None:
        self._utils.check_empty_binary_tree()
        self._utils.red_black_is_sentinel(node)
        self._utils.validate_tree_node(node, RedBlackNode)
        while node.right != self.NIL:
            node = node.right
        return node

    def predecessor(self, node) -> iBSTNode[T, K] | None:
        """predecessor = next key less than current key -- can return a sentinel"""
        self._utils.check_empty_binary_tree()
        self._utils.red_black_is_sentinel(node)
        self._utils.validate_tree_node(node, RedBlackNode)

        # Case 1: Node has left child --traverse down tree
        if node.left != self.NIL:
            current_node = node.left  # 1 time
            while current_node.right != self.NIL:
                current_node = current_node.right
            return current_node  # last node

        # Case 2: Node has no left child -- climb up tree
        current_node = node
        parent_node = current_node.parent
        # this means -traverse up while the current node is less than the parent
        while parent_node != self.NIL and current_node == parent_node.left:
            current_node = parent_node
            parent_node = parent_node.parent
        return parent_node  # can be NIL

    def successor(self, node) -> iBSTNode[T, K] | None:
        """
        successor = next key greater than current key -- can return a sentinel
        """
        self._utils.check_empty_binary_tree()
        self._utils.red_black_is_sentinel(node)
        self._utils.validate_tree_node(node, RedBlackNode)

        # Case 1: Node has right child -- traverse down tree
        if node.right != self.NIL:
            current_node = node.right  # go right 1 time.
            while current_node.left != self.NIL:
                current_node = current_node.left
            return current_node  # return last node from left subtree
        # Case 2: Node has no right child -- climb up tree
        current_node = node
        parent_node = current_node.parent
        # climb tree whle current node is greater than parent.
        while parent_node != self.NIL and current_node == parent_node.right:
            # traverse up tree
            current_node = parent_node
            parent_node = parent_node.parent
        return parent_node  # can be None if node is max key.

    def find_lower_bounds(self, key):
        """
        returns the smallest node with a key >= to the specifed key.
        the input key itself does not have to Exist in the red black tree.
        can return a Sentinel (self.NIL)
        """

        current = self._root
        candidate = self.NIL

        while current is not self.NIL:
            if key <= current.key:
                candidate = current
                current = current.left
            else:
                current = current.right
        
        return candidate

    def height(self) -> Index:
        return self._utils.red_black_tree_height()

    def search(self, key) -> iBSTNode[T, K] | None:
        self._utils.check_empty_binary_tree()
        key = Key(key)
        self._utils.check_key_is_same_type(key)
        return self._utils.red_black_descent(self._root, RedBlackNode, key)

    # ----- Mutators -----
    def simple_bst_insert(self, key, value):
        """insert via bst insertion protocol"""
        # initialize variables for traversal
        parent_node = self.NIL
        current_node = self._root
        # for upsert case -- we need to travel the whole tree to avoid red property violations
        is_upsert = False 
        upserted_node = self._root

        # traverse Tree
        while current_node != self.NIL:
            parent_node = current_node
            # * Upsert Case: if a match is found - update value and return node.
            if key == current_node.key:
                current_node.element = value
                is_upsert = True
                return current_node, is_upsert
            # if key is smaller - traverse left
            if key < current_node.key:
                current_node = current_node.left
            # if key is larger - traverse right
            else:
                current_node = current_node.right

        # * Create a new Node (always colored Red) and sentinels for the children
        new_node = RedBlackNode(self.datatype, key, value, sentinel=self.NIL, node_colour=NodeColor.RED, tree_owner=self)
        new_node.set_red()
        new_node.left = new_node.right = self.NIL
        # connect new node to parent (at the end of the tree)
        new_node.parent = parent_node

        # * root Case: tree is empty
        if parent_node is self.NIL:
            self._root = new_node
        # * smaller key - insert left child
        elif key < parent_node.key:
            parent_node.left = new_node
        # * larger key - insert right child
        else:
            parent_node.right = new_node
        return new_node, is_upsert

    def insert(self, key, value) -> iBSTNode[T, K]:
        """
        Insert Node into red black tree
        """
        value = TypeSafeElement(value, self.datatype)
        key = Key(key)
        self._utils.set_keytype(key)
        self._utils.check_key_is_same_type(key)
        new_node, is_upsert = self.simple_bst_insert(key, value)
        if not is_upsert:
            self._utils.repair_red_property(new_node)
            self._utils.check_red_children_are_black(RedBlackNode)
            # print(self._utils.debug_insertion_print(new_node))
        return new_node

    def replace(self, node, value) -> T:
        """replace element value of specified node. (structure doesnt change)"""
        self._utils.validate_tree_node(node, RedBlackNode)
        value = TypeSafeElement(value, self.datatype)
        old_value = node.element
        node.element = value # type: ignore
        return old_value

    def delete(self, node) -> T:
        """deletes a node from the red black tree (similar to BST delete) and repairs the black property of a red black tree"""
        # empty tree case
        self._utils.check_empty_binary_tree()
        self._utils.red_black_is_sentinel(node)
        self._utils.validate_tree_node(node, RedBlackNode)

        # save the original color of the target node
        old_value = node.element
        old_node = node
        original_color = old_node.color
        node.alive = False
        node.tree_owner = None
        # print(f"\nDeleting: {old_value} [{original_color}]")

        # * 1 Child Case: auto handles 0 child leaf case
        # discover children and replace target node with its child
        if node.left == self.NIL:
            # the node that moved into node’s position (can be NIL)
            replacement = node.right
            self._utils.transplant(node, node.right)  # this replaces the target node
        # same for right child
        elif node.right == self.NIL:
            replacement = node.left
            self._utils.transplant(node, node.left)

        # * 2 Child Case:
        else:
            # * initialze successor
            succ = self.successor(node)
            original_color = succ.color
            node.key = succ.key
            node.element = succ.element
            # occupies succ original node pos once succ swaps with target node
            replacement = succ.right
            # * Case A: successor is direct right child of node:
            # we pre-emptively relink replacement to succ. as succ is about to be replaced
            if succ.parent == node:
                replacement.parent = succ   # replacement parent is now the successor
            # * Case B: succ is deeper down in the right subtree:
            else:
                # Step 1: remove successor from old spot
                self._utils.transplant(succ, succ.right)
                # Step 2: attach the entire right subtree of node to succ child
                succ.right = node.right
                # Step 3: fix parent pointer
                succ.right.parent = succ

            # * replaces the target node with the successor
            self._utils.transplant(node, succ)
            # relinks original left child to successor
            succ.left = node.left
            # updates parent link of new succ left child
            succ.left.parent = succ
            # inherits the original color from target node
            succ.color = old_node.color

        # * if deleted node was black - run repair black violation
        if original_color == NodeColor.BLACK:
            self._utils.repair_black_property(replacement)
            # print(f"Physical Nodes: {self._utils.debug_count_real_nodes(RedBlackNode)}")
            # print(f"Total Nodes: {len(self)}\n")
        # assertions & property violation guards:
        self._utils.check_red_children_are_black(RedBlackNode)
        assert self._root.color == NodeColor.BLACK, f"The root must always be black after deletion"
        # self._utils.black_height_debug_print()
        # assert self.is_black_property, f"Number of Black Nodes for any path must be equal."
        return old_value

    # ----- Traversals -----
    def preorder(self):
        return self._utils.red_black_dfs_traversal(self._root, RedBlackNode)

    def postorder(self):
        return self._utils.red_black_postorder_traversal(self._root, RedBlackNode)

    def levelorder(self):
        return self._utils.red_black_bfs_traversal(self._root, RedBlackNode)

    def inorder(self):
        return self._utils.red_black_inorder_traversal(self._root, RedBlackNode)


# Main ----------- Client Facing Code ------------

# todo stress test with 100s of items.

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

    rb = RedBlackTree(str)
    print(rb)
    print(repr(rb))
    print(f"\nTesting Is_empty?: {rb.is_empty()}")

    random_keys = [i for i in range(100)]
    key_sample = random.sample(random_keys, 30)

    print(f"\nTesting Insertion: ")
    for keys, data in zip(key_sample, random_data):
        rb.insert(keys, data)
    print(rb)
    # print(repr(rb))
    # print(f"Inorder Traversal: {[i.element for i in rb.inorder()]}")

    print(f"\nTesting Upsert:")
    keyhhs = [5, 3, 8, 3, 5, 7]
    for k in keyhhs:
        rb.insert(k, f"VALUE {k}")
    # print(rb)
    # print(repr(rb))

    print(f"\nTesting Is_empty?: {rb.is_empty()}")
    the_root = rb.root
    max = rb.maximum(rb.root)
    min = rb.minimum(rb.root)
    succ_of_min = rb.successor(min)
    pred_of_max = rb.predecessor(max)

    print(f"\nTesting root: {the_root.element} Type: {type(the_root).__name__}")
    print(f"Testing Max: {max.element}")
    print(f"Testing Min: {min.element}")
    print(f"Testing successor of Min: {succ_of_min.element}")
    print(f"Testing predecessor of Max: {pred_of_max.element}")

    print(f"\nTesting replace functionality: replacing {the_root.element}")
    old_value = rb.replace(the_root, "THE ROOT")
    print(f"replaced: {old_value}. New value={the_root.element}")
    print(rb)

    print(f"\nTesting Deletion on a subset of items")
    keys_list = [i for i in rb.inorder()]
    keys_subset = keys_list[:10]
    print(f"items to delete: {len(keys_subset)}")
    print(f"Items: {', '.join([i.element for i in keys_subset])}")
    print(f"Total Nodes: {len(rb)}")
    for i in keys_subset:
        rb.delete(i)
    print(rb)
    print(f"Is the item there?: {[i.element for i in rb.inorder()]}")


if __name__ == "__main__":
    main()
