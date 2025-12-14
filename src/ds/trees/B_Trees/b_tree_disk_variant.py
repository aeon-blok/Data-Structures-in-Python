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
import pickle
import os
import pathlib

# endregion

# region custom imports
from user_defined_types.generic_types import (
    T,
    ValidDatatype,
    TypeSafeElement,
    Index,
    ValidIndex,
)

from utils.validation_utils import DsValidation
from utils.representations import BTreeNodeRepr, BTreeRepr
from utils.helpers import RandomClass
from utils.exceptions import *

from adts.collection_adt import CollectionADT
from adts.b_tree_adt import BTreeADT

from ds.sequences.Stacks.array_stack import ArrayStack
from ds.primitives.arrays.dynamic_array import VectorArray, VectorView
from ds.trees.tree_nodes import BTreeNode
from ds.trees.tree_utils import TreeUtils

from user_defined_types.generic_types import Index, ValidDatatype, ValidIndex, TypeSafeElement, PositiveNumber
from user_defined_types.key_types import iKey, Key
from user_defined_types.tree_types import NodeColor, Traversal

# endregion

"""
B Tree:
The reason we store so many keys in a node in a b tree - is due to the differences of RAM Memory and Disk Memory.
Because disk read & write is so much slower. comparing keys is more efficient than searching disk blocks.

"""


class BTree(BTreeADT[T], CollectionADT[T], Generic[T]):
    """
    B Tree Data Structure Implementation:
    Duplicate Keys are forbidden.
    Utilizes Pre-emptive fix Strategy for insert and deletion. (CLRS)
    Stores key and element values.
    Traversal solely uses inorder traversal - (can output keys, values or tuples of both)
    """
    def __init__(self, datatype: type, degree: int) -> None:
        self._datatype = ValidDatatype(datatype)
        self._tree_keytype: None | type = None
        self._root: None | BTreeNode = None
        self._degree = PositiveNumber(degree)
        self._total_nodes: int = 0
        self._total_keys: int = 0

        # composed objects
        self._utils = TreeUtils(self)
        self._validators = DsValidation()
        self._desc = BTreeRepr(self)

    @property
    def datatype(self) -> type:
        return self._datatype

    @property
    def tree_keytype(self) -> None | type:
        return self._tree_keytype

    @property
    def total_keys(self) -> int:
        """returns the total number of keys in the b tree"""
        return self._total_keys

    @property
    def tree_height(self) -> int:
        """the max tree height of the btree"""
        return self._utils.btree_height_iterative(BTreeNode)

    @property
    def validate_tree(self) -> None:
        self._utils.validate_btree()

    @property
    def bfs_view(self):
        return self._utils.b_tree_bfs_view(BTreeNode)

    @property
    def max_keys(self) -> int:
        """2t-1 -- defines the maximum number of keys allowed per node, derived from the degree."""
        return (2 * self._degree) - 1

    @ property
    def min_keys(self) -> int:
        """t-1 -- defines the minimum number of keys allowed per node, derived from degree"""
        return self._degree - 1

    @property
    def root(self):
        return self._root

    @property
    def total_nodes(self) -> int:
        return self._total_nodes

    # ----- Meta Collection ADT Operations -----
    def is_empty(self) -> bool:
        return self._root is None

    def clear(self) -> None:
        """iteratively deletes all the nodes and resets counters etc."""
        self._root = None
        self._total_keys = 0
        self._total_nodes = 0

    def __contains__(self, key) -> bool:
        return self.search(key) is not None

    def __len__(self) -> Index:
        return self._total_keys

    def __iter__(self) -> Generator[T, None, None]:
        """returns the element value via inorder traversal (smallest to largest key)"""
        for k,v in self.traverse(return_type='tuple'):
            yield v

    def __reversed__(self) -> Generator[T, None, None]:
        """reverses iteration - largest to smallest element is returned."""
        elements = self.traverse('elements')
        for v in reversed(elements):
            yield v

    # ----- Utilities -----
    def __str__(self) -> str:
        return self._desc.str_btree()

    def __repr__(self) -> str:
        return self._desc.repr_btree()

    def __bool__(self):
        return self._total_nodes > 0

    # ----- Canonical ADT Operations -----
    # ----- Accessors -----

    # todo implement binary search

    def _recursive_search(self, node: BTreeNode, key) -> Optional[tuple[BTreeNode, int]]:
        """recursively searches the whole tree until a match is found or None is returned."""
        # * empty tree case: existence check

        # init vars
        current_node = node
        idx = 0

        # * traverse all the keys in the node.
        # If the key we are searching for is greater than the current index. continue traversal.
        while idx <= current_node.num_keys -1 and key > current_node.keys[idx]:
            idx += 1  # increment counter

        # * key match. - return a tuple of the node and index.
        if idx <= current_node.num_keys -1 and key == current_node.keys[idx]:
            return (current_node, idx)
        # * key not found
        elif current_node.is_leaf:
            return None
        # * search child nodes -- the key must be in child[idx] - due to the b tree children property (the ordering of the keys and children)
        else:
            child = node.children[idx]
            return self._recursive_search(child, key)

    def _node_search(self, key) -> Optional[tuple[BTreeNode, int]]:
        """
        Searches by key for the node that contains the key. 
        returns a tuple of the node and the key index. which can be accessed via the node.
        """
        return self._recursive_search(self._root, key)

    def search(self, key) -> T | None:
        """
        public facing method
        Searches for the specified key in the B tree and returns the element value.
        """

        # *empty tree case
        if self._root is None:
            return None

        # validate key
        key = Key(key)
        self._utils.check_btree_key_is_same_type(key)

        node_and_index = self._node_search(key)
        if node_and_index is not None:
            node, idx = node_and_index
            key = node.keys[idx]
            element: T = node.elements[idx]
            return element
        else:
            return None

    def _predecessor(self, node: BTreeNode) -> tuple[BTreeNode, int]:
        """returns the predecessor key - that is the largest key in the left subtree smaller than the specified key."""
        current = node
        while not current.is_leaf:
            # traverse to the rightmost child.
            current = current.children[current.num_keys - 1]
        return (current, current.num_keys - 1)

    def _successor(self, node: BTreeNode) -> tuple[BTreeNode, int]:
        """returns the succesor key  - the smallest key in the right subtree lareger than the specified key."""
        current = node
        while not current.is_leaf:
            current = current.children[0]
        return (current, 0)

    def min(self) -> Optional[T]:
        """returns the minimum element in the b tree"""
        current = self._root
        # empty tree case:
        if current is None: return None

        # traverse
        while not current.is_leaf:
            current = current.children[0]

        element: T = current.elements[0]
        return element

    def max(self) -> Optional[T]:
        """returns the max key (paired element) in the b tree"""
        current = self._root
        # empty tree case:
        if current is None: return None

        # traverse
        while not current.is_leaf:
            current = current.children[current.num_keys]

        last = current.num_keys - 1
        element: T = current.elements[last]
        return element

    # ----- Mutators -----
    def create_tree(self) -> None:
        """Creates a B tree and the root node"""
        self._root = BTreeNode(self._datatype, self._degree, is_leaf=True)
        self._total_nodes +=1

    def _insert_non_full(self, node, key, value):
        """
        helper method: inserts into a non full node.
        """
        # the last key index
        idx = node.num_keys - 1

        # * leaf case: - insert key into node. (no further action needed)
        if node.is_leaf:
            # shifting keys to make room for new key.
            while idx >= 0 and key < node.keys[idx]:
                idx -= 1
            # insert key and value into the node
            node.keys.insert(idx+1, key)
            node.elements.insert(idx+1, value)
            self._total_keys += 1

        # * internal node - find the child where key belongs
        else:
            # traverse backwards through keys until new key is greater than current key
            while idx >= 0 and key < node.keys[idx]:
                idx -= 1
            # move forward 1 step to get the correct index for the new key.
            idx += 1
            # * split child if its full
            if node.children[idx].num_keys == self.max_keys:
                self.split_child(node, idx)
                # if new key is larger -- it goes in the right child. otherwise goes in the left child.
                if key > node.keys[idx]:
                    idx += 1
            # insert key and value into the correct child slot.
            self._insert_non_full(node.children[idx], key, value)

    def insert(self, key, value: T) -> None:
        """
        Public Facing Insert Method: Inserts a Key Value Pair into an existing leaf node.
        Overflow Rule: If the node is full - performs a split child/root operation. (on every full node you encounter traversing the tree.)
        Fix Then Insert Strategy: Utilizes the strategy employed by CLRS - 
        Nodes are pre-emptively checked for number of keys and split if full. 
        this allows the insertion to be completed in a single traversal down the tree. 
        rather than having to go back up the tree to fix nodes that violate the b tree properties.
        """

        key = Key(key)
        self._utils.check_btree_key_is_same_type(key)
        value = TypeSafeElement(value, self._datatype)

        # *empty tree case: insert into root node.
        if self._root is None:
            self._root = BTreeNode(self._datatype, self._degree, is_leaf=True)
            self._total_nodes += 1
            self._insert_non_full(self._root, key, value)
            return

        # * root is full
        if self._root.num_keys == self.max_keys:
            new_root = self.split_root()
            self._insert_non_full(new_root, key, value)
        # * insert into the root if not full.
        else:
            self._insert_non_full(self._root, key, value)

    def delete(self, key) -> None:
        """
        public delete method - utilizes recursive deletion.
        Fix then Delete Strategy: Utilizes pre-emptive checking to ensure that every child has over the min number of keys. 
        which allows us to delete a key without extra operations.
        If they do not have the required number of keys (t) then perform a borrow or merge operation
        """

        key = Key(key)
        self._utils.check_btree_key_is_same_type(key)
        print(f"\nB-tree delete: {key}")
        # * Empty tree Case:
        if self._root is None:
            print(f"btree is empty - no further action")
            return

        self._recursive_delete(self._root, key)

        # * root edge case:
        if self._root.num_keys == 0:
            if not self._root.is_leaf:
                # promote only child to be new root.
                self._root = self._root.children[0]
                self._total_nodes -= 1
            else:
                # tree is empty:
                self._root = None
                self._total_nodes -= 1

    def _case_1_leaf_node_contains_key(self, parent_node, idx):
        """ Case 1A: current has min + 1 keys:"""
        print(f"CASE 1: Entering Case 1")
        if parent_node.num_keys > self.min_keys:
            print(f"Deleting Key: {parent_node.keys[idx]}")
            parent_node.keys.delete(idx)
            parent_node.elements.delete(idx)
            self._total_keys -= 1
        elif parent_node == self._root:
            print(f"ROOT CASE: Node is the Root and the only node left: deleting Key: {parent_node.keys[idx]}")
            parent_node.keys.delete(idx)
            parent_node.elements.delete(idx)
            self._total_keys -= 1
        else:
            raise KeyInvalidError(f"Error: Case 1: Key not found.")

    def _case_2_internal_node_contains_key(self, parent_node, idx, key):
        """
        Case 2A: child i has the min + 1 required keys
        Case 2B: child i has min keys, and its right sibling has min + 1 keys
        Case 2C: both child and sibling have min keys. (cant borrow need to merge.)
        """
        child = parent_node.children[idx]
        right_sibling = parent_node.children[idx+1] if idx + 1 < parent_node.num_keys + 1 else None
        left_sibling = parent_node.children[idx - 1] if idx > 0 else None

        if child.num_keys >= self._degree:
            print(f"CASE 2A: Entering Case 2A: child pointer={child}")
            # find predecessor:
            pred, pred_idx = self._predecessor(child)
            pred_key: iKey = pred.keys[pred_idx]
            print(f"predecessor: {pred_key} and {pred}")
            # replace parent key with predecessor key.
            parent_node.keys[idx] = pred_key
            assert child.num_keys >= self._degree, f"Error: Case 2A: Child doesnt have t keys."
            print(f"Case 2A: recursively entering child with pred key")
            self._recursive_delete(child, pred_key)
            return
        
        elif child.num_keys == self.min_keys and right_sibling is not None and right_sibling.num_keys >= self._degree:
            print(f"CASE 2B: Entering Case 2B: child pointer={child}, right sibling={right_sibling}")
            # find successor:
            succ, succ_idx = self._successor(right_sibling)
            succ_key = succ.keys[succ_idx]
            print(f"succesor: {succ_key}, {succ}")
            # replace parent key with succ key
            parent_node.keys[idx] = succ_key
            assert right_sibling.num_keys >= self._degree, f"Error: Case 2B: Child doesnt have t keys."
            print(f"Case 2B: recursively entering right sibling with succ key")
            self._recursive_delete(right_sibling, succ_key)
            return
        
        # * Case 2C: both child i and siblings have min keys. (cant borrow need to merge.)
        elif child.num_keys == self.min_keys: 
            print(f"CASE 2C: Entering Case 2C child={child}, right={right_sibling}, left={left_sibling}")
            # merge right sibling into child
            if right_sibling is not None and right_sibling.num_keys == self.min_keys:
                self._total_nodes -= 1
                print(f"merge right into child operation:")
                self.merge_right_into_child(parent_node, idx)
                merged_child = parent_node.children[idx]
                print(f"merged={merged_child}")
                assert merged_child.num_keys == self.max_keys, f"Error: Case 2C: Merged Child should have Max number of keys. (CLRS)"
                assert merged_child.num_keys >= self._degree, f"Error: Case 2C: Child doesnt have t keys."
                print(f"Entering recursive delete on merged child.")
                self._recursive_delete(merged_child, key)
                return
            # * Last Child Edge Case: merge child into left sibling (affects index order)
            elif left_sibling is not None and left_sibling.num_keys == self.min_keys:
                self._total_nodes -= 1
                self.merge_with_left(parent_node, idx)
                print(f"merge child with left operation:")
                merged_node = parent_node.children[idx-1]
                print(f"merged={merged_node}")
                assert merged_node.num_keys == self.max_keys, f"Error: Case 2C: Merged left sibling should have Max number of keys. (CLRS)"
                assert merged_node.num_keys >= self._degree, f"Error: Case 2C: left sibling doesnt have t keys."
                print(f"Entering recursive delete on merged node.")
                self._recursive_delete(merged_node, key)
                return
            else:
                raise NodeExistenceError(f"Error: Case 2C: sibling must have min keys. B Tree property violated")
        else:
            raise NodeExistenceError(f"Error: Case 2 Entered but cannot satisfy invariants.")

    def _case_3_internal_node_does_not_contain_key(self, parent_node, idx, key):
        """
        Case 3A: Child i has min (t-1) keys, but sibling has min + 1 keys -- (borrow from sibling)
        Borrow median key from parent. and swap this with sibling.
        Case 3B:  Child and siblings have min keys (merge child with sibling)
        we need to move a key from current node to become the median key for this new merged node.
        Merging with right is preferable because it maintains index order.
        """

        # init family unit
        child = parent_node.children[idx]
        left_sibling = parent_node.children[idx - 1] if idx > 0 else None
        right_sibling = parent_node.children[idx+1] if idx + 1 < parent_node.num_keys + 1 else None
        print(f"CASE 3: entering case 3: child={child}, left={left_sibling}, right={right_sibling}")
        if child.num_keys == self.min_keys:
            # * Case 3A: Child i has min keys, but sibling has min + 1 keys -- (borrow from sibling)
            # Case 3A: borrow key from left sibling
            if left_sibling is not None and left_sibling.num_keys > self.min_keys:
                print(f"Case 3A: borrow from left. performing borrow left op")
                self.borrow_left(parent_node, idx)
                child = parent_node.children[idx]
                print(f"child={child} Entering recursive delete on child")
                self._recursive_delete(child, key)

            # Case 3A: borrow key from right sibling
            elif right_sibling is not None and right_sibling.num_keys > self.min_keys:
                print(f"Case 3A: borrow from right. performing borrow right op")
                self.borrow_right(parent_node, idx)
                child = parent_node.children[idx]
                print(f"child={child} Entering recursive delete on child")
                self._recursive_delete(child, key)

            # * Case 3B:  Child and siblings have min keys (merge child with sibling)
            elif right_sibling is not None and right_sibling.num_keys == self.min_keys:
                print(f"Case 3B: Merge Right -- performing merge right into child op")
                self._total_nodes -= 1
                self.merge_right_into_child(parent_node, idx)
                merged_child = parent_node.children[idx]
                assert merged_child.num_keys == self.max_keys, f"Error: Case 3B: Merged Child should have Max number of keys. (CLRS)"
                print(f"merged child={merged_child} Entering recursive delete on merged child")
                self._recursive_delete(merged_child, key)

            # merge with left sibling (if it exists.)
            elif left_sibling is not None and left_sibling.num_keys == self.min_keys:
                print(f"Case 3B: Merge Left -- performing merge child into left op")
                self._total_nodes -= 1
                self.merge_with_left(parent_node, idx)
                merged_node = parent_node.children[idx-1]
                assert merged_node.num_keys == self.max_keys, f"Error: Case 3B: Merged Node (left sibling) should have Max number of keys. (CLRS)"
                print(f"merged child={merged_node} Entering recursive delete on merged child")                
                self._recursive_delete(merged_node, key)

            # In a properly formed B-tree (degree ≥ 2), this should never trigger, but it catches any logic/invariant violation.
            else:
                raise NodeExistenceError(f"Error: Case 3B: Invariant violated - Child does not have a sibling.")
        else:
            print(f"Child didnt have min keys.... Traversing to child and deleting.")
            self._recursive_delete(child, key)

    def _recursive_delete(self, node: BTreeNode, key: iKey) -> None:
        """
        Underflow Rule: if deletion causes a node to have less than t-1 keys - performs a merge, or borrow operation to rebalance.
        Deletion Method is designed in a way that we ensure that every recursive call on a node ensures that the node has the minimum number of required keys.
        Utilizes a Pre-emptive rebalancing strategy (CLRS method)
        Case 1: Leaf Node contains key
        Case 2: Internal Node contains key
        Case 3: Internal Node does not contain key
        """
        # init vars
        idx = 0
        parent_node = node
        if parent_node == self._root:
            print(f"Entering Recursive Delete on Root: ROOT={parent_node} is_leaf: {node.is_leaf}")
        else:
            print(f"Entering Recursive Delete: node={parent_node} is_leaf: {node.is_leaf}")

        # * Linear Scan: traverse through keys and find the key...
        while idx < parent_node.num_keys and key > parent_node.keys[idx]:
            idx += 1  # increment counter
        print(f"Linear Scan Finished on {idx}/{parent_node.num_keys}")

        # * Case 1: Leaf Node Contains Key: delete immmediately (only if it has > min keys)
        if parent_node.is_leaf:
            if idx < parent_node.num_keys and key == parent_node.keys[idx]:
                self._case_1_leaf_node_contains_key(parent_node, idx)
            return

        # * Case 2: Internal node contains the key.
        if idx < parent_node.num_keys and key == parent_node.keys[idx] and not parent_node.is_leaf:
            self._case_2_internal_node_contains_key(parent_node, idx, key)
            return

        # * Case 3: Internal Node does not contain key
        # essentially this works like a pre-emptive check -- enforcing that every child has over the min required keys.

        if not parent_node.is_leaf and key not in parent_node.keys:
            self._case_3_internal_node_does_not_contain_key(parent_node, idx, key)

    # ----- Traversal -----
    def traverse(self, return_type: Literal['keys', 'elements', 'tuple']) -> Iterable:
        """
        Traverse throughout the B Tree and return a sequence of all the kv pairs in the tree. 
        In a specifed order (inorder)
        """
        keys = VectorArray(self._total_keys, object)
        elements = VectorArray(self._total_keys, self._datatype)
        tuples = VectorArray(self._total_keys, tuple)

        for i in self._utils.b_tree_inorder():
            k, v = i
            keys.append(k.value) # unpack the key() object
            elements.append(v)
            tuples.append((k.value, v))

        if return_type == 'keys':
            return keys

        if return_type == 'elements':
            return elements

        if return_type == 'tuple':
            return tuples

    # ----- Utility -----
    def split_root(self):
        """
        splits the root node: this is the only way to increase the height of a B Tree
        creates a new node, the old root is parented to the new root
        new root is linked to its child the old root.
        then we split the child (the old root)
        and return the new root node.
        """
        new_node = BTreeNode(self._datatype, self._degree, is_leaf=False)
        self._total_nodes += 1
        # make the old root a child of the new node.
        new_node.children.insert(0, self._root)
        # new node becomes the new root.
        self._root = new_node
        # Split the first child of new_node, which is the old root
        self.split_child(new_node, 0)
        return new_node

    def split_child(self, parent_node: BTreeNode, index: Index) -> None:
        """
        split the full node into 2 different nodes.
        We split via the median key
        all nodes > median go to the new right node,
        all < median go to the left node.
        promote the median key up to the parent.
        remove median key from child
        indices: 0 … t-2 | t-1 | t … 2t-2      
                left      median    right
        """
        # child - retains the first half of the keys
        child_node: BTreeNode = parent_node.children[index]

        # * we create a new sibling - it will inherit its leaf status from its other sibling (the child)
        new_sibling: BTreeNode = BTreeNode(self._datatype, self._degree, is_leaf=child_node.leaf)

        self._total_nodes += 1

        median_key =  child_node.keys[self._degree - 1]
        median_element = child_node.elements[self._degree - 1]

        # * collect the largest keys and elements from the child. and give them to the sibling.
        # moves the minimum number of keys necessary to the new node
        for idx in range(self.min_keys):
            # copies over the keys that are higher than the min number of keys.
            new_sibling.keys.append(child_node.keys[idx + self._degree])
            new_sibling.elements.append(child_node.elements[idx + self._degree])
        # copy over children also
        if not child_node.is_leaf:
            for idx in range(self._degree):
                new_sibling.children.append(child_node.children[idx + self._degree])

        # * delete the second half of keys and children from child node.
        for _ in range(self.min_keys):
            child_node.keys.delete(self._degree)
            child_node.elements.delete(self._degree)
        if not child_node.is_leaf:
            for _ in range(self._degree):
                child_node.children.delete(self._degree)

        # * relink parent and new child. (and add promoted key)
        # add new sibling to parent's children list
        parent_node.children.insert(index+1, new_sibling)

        # now insert promoted median key. (t-1)
        parent_node.keys.insert(index, median_key)
        parent_node.elements.insert(index, median_element)

        # remove median key from child node.
        child_node.keys.delete(self._degree-1)
        child_node.elements.delete(self._degree-1)

    def merge_right_into_child(self, parent_node, idx: Index) -> None:
        """
        Merges the right sibling into the child node.
        the child is removed from the parent children list.
        """
        child = parent_node.children[idx]
        right_sibling = parent_node.children[idx+1]

        # move median key down from parent
        median_key = parent_node.keys[idx]
        median_element = parent_node.elements[idx]
        child.keys.append(median_key)
        child.elements.append(median_element)

        # merge right sibling INTO child
        for i in right_sibling.keys: child.keys.append(i)
        for i in right_sibling.elements: child.elements.append(i)
        for i in right_sibling.children: child.children.append(i)

        # remove median key / element from parent
        parent_node.keys.delete(idx)
        parent_node.elements.delete(idx)

        # remove right sibling from parent
        parent_node.children.delete(idx + 1)

    def merge_with_left(self, parent_node, idx: Index) -> None:
        """
        Merges a child node into its left sibling. for this it uses its parent's node's median key. (its passed down)
        the child is then removed from the parent...
        """
        child = parent_node.children[idx]
        left_sibling = parent_node.children[idx - 1]

        # move parent key down to become median key for new merged node.
        median_key = parent_node.keys[idx - 1]
        median_element = parent_node.elements[idx - 1]
        # append median key to the array.
        left_sibling.keys.append(median_key)
        left_sibling.elements.append(median_element)

        # now append the child keys INTO the Left sibling. and elements.
        for i in child.keys: left_sibling.keys.append(i)
        for i in child.elements: left_sibling.elements.append(i)
        for i in child.children: left_sibling.children.append(i)

        # * delete median key / element from parent.
        parent_node.keys.delete(idx-1)
        parent_node.elements.delete(idx-1)

        # remove child from parent.
        parent_node.children.delete(idx)

    def borrow_left(self, parent_node, idx: Index) -> None:
        """
        Borrows the last key / element from the left sibling and moves it up to the parent.
        then moves the corresponding parent key / element down to the RIGHT child
        assumes the nodes involved are internal nodes.
        The key separating the two nodes is at index (idx - 1)
        Borrow is in essence a rotation, applied to two keys.
        """

        child = parent_node.children[idx]
        left_sibling = parent_node.children[idx-1]

        # move parent key down into child:
        child.keys.insert(0, parent_node.keys[idx-1])
        child.elements.insert(0, parent_node.elements[idx-1])

        # move last key from left sibling up into parent
        last = left_sibling.num_keys - 1    # is this correct?
        parent_node.keys[idx-1] = left_sibling.keys[last]
        parent_node.elements[idx-1] = left_sibling.elements[last]

        # move child pointer from left sibling to child children array.
        if not left_sibling.is_leaf:
            child.children.insert(0, left_sibling.children[last])
            left_sibling.children.delete(last)

        # delete key from left sibling
        left_sibling.keys.delete(last)
        left_sibling.elements.delete(last)

    def borrow_right(self, parent_node, idx: Index) -> None:
        """
        Borrows the first key / element from the right sibling and moves it up to the parent.
        Then Moves the Corresponding parent key / element down into the LEFT child.
        parent key --> child key.
        right_sibling key --> parent key
        Borrow is in essence a rotation, applied to two keys.
        """
        child = parent_node.children[idx]
        right_sibling = parent_node.children[idx+1]

        # move key from parent down into child
        child.keys.append(parent_node.keys[idx])  #maybe idx
        child.elements.append(parent_node.elements[idx])

        # move first key from right sibling up into parent
        parent_node.keys[idx] = right_sibling.keys[0]
        parent_node.elements[idx] = right_sibling.elements[0]

        # move child pointer from right sibling to child children array.
        # ONLY if internal node. (leaves dont have children)
        if not right_sibling.is_leaf:
            child.children.append(right_sibling.children[0])
            right_sibling.children.delete(0)

        # delete first key from right sibling.
        right_sibling.keys.delete(0)
        right_sibling.elements.delete(0)


# ------------------------------- Main: Client Facing Code: -------------------------------
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

    keys = [i for i in range(len(random_data))]

    b = BTree(str, 5)
    print(f"Does key 3 exist? {'Yes' if 3 in b else 'No'}")

    print(f"\nTesting Insert functionality of Btree")
    for i, item in zip(keys, random_data):
        b.insert(i, item)

    print(repr(b))
    print(b)
    b.validate_tree

    print(f"\nTesting Node repr")
    print(b._root)
    print(repr(b.root))

    print(f"\nTesting Search Functionality: key:25 = {b.search(25)}")
    print(f"Testing Search on a non existent key: key:200 = {b.search(200)}")

    min_val = b.min()
    max_val = b.max()
    print(f"Min element: {min_val}")
    print(f"Max element: {max_val}")

    print("\nTesting __contains__ and __len__...")
    print(f"Does key 3 exist? {'Yes' if 3 in b else 'No'}")
    print(f"Total keys in tree: {len(b)}")
    
    # ---------- Traverse ----------
    print("\nTesting traversal...")
    print(b.traverse("keys"))
    print(b.traverse("elements"))
    print(b.traverse("tuple"))
  
    print(f"\nTesting Delete functionality...")
    print(b)
    b.validate_tree
    print(f"Testing randomized deletion")
    shuffled_keys = list(range(30))
    random.shuffle(shuffled_keys)
    for key in shuffled_keys:
        b.delete(key)
        print(b)
    b.validate_tree

    # ---------- Type Checking ----------
    print("\nTesting type validation...")
    try:
        b.insert(6, RandomClass("alyyyllgfdgfd"))  # invalid element type
    except Exception as e:
        print(f"Caught expected type error: {e}")


if __name__ == "__main__":
    main()
