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
    """
    def __init__(self, datatype: type, degree: int, traversal: Traversal = Traversal.INORDER) -> None:
        self._datatype = ValidDatatype(datatype)
        self._keytype: None | type = None
        self._root: None | BTreeNode[T] = None
        self._degree = PositiveNumber(degree)
        self._traversal = traversal
        self._total_nodes: int = 0

        # composed objects
        self._utils = TreeUtils(self)
        self._validators = DsValidation()
        self._desc = BTreeRepr(self)

    @property
    def datatype(self) -> type:
        return self._datatype

    @property
    def traversal(self):
        return self._traversal

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
        return super().clear()

    def __contains__(self, value: T) -> bool:
        return super().__contains__(value)

    def __len__(self) -> Index:
        return self._total_nodes

    def __iter__(self):
        pass

    def __reversed__(self):
        pass

    # ----- Utilities -----
    def __str__(self) -> str:
        return self._desc.str_btree()

    def __repr__(self) -> str:
        return self._desc.repr_btree()

    def __bool__(self):
        return self._total_nodes > 0

    # ----- Canonical ADT Operations -----
    # ----- Accessors -----
    def search_for_element(self, key) -> T | None:
        """Searches for the specified key in the B tree and returns the value."""
        node, idx = self.search(key)
        element = node.elements[idx]
        return element

    # todo implement binary search

    def _recursive_search(self, node, key) -> Optional[tuple[BTreeNode[T], int]]:
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

    def search(self, key) -> Optional[tuple]:
        """Searches by key for the node that contains the key. returns a tuple of the node and the key index. which can be accessed via the node."""

        return self._recursive_search(self._root, key)

    def _predecessor(self, node, index):
        """returns the predecessor key - that is the largest key in the left subtree smaller than the specified key."""
        child = node.children[index]
        while not child.is_leaf:
            # traverse to the rightmost child.
            child = child.children[child.num_keys-1]
        return (child, child.num_keys-1)

    def _successor(self, node, index):
        """returns the succesor key  - the smallest key in the right subtree lareger than the specified key."""
        child = node.children[index]
        while not child.is_leaf:
            child = child.children[0]
        return (child, 0)

    def min(self) -> Optional[tuple]:
        """returns the minimum key (& paired element) in the b tree"""
        current = self._root
        # empty tree case:
        if current is None: return None

        # traverse
        while not current.is_leaf:
            current = current.children[0]

        return (current.keys[0], current.elements[0])

    def max(self):
        """returns the max key (paired element) in the b tree"""
        current = self._root
        # empty tree case:
        if current is None: return None
        
        # traverse
        while not current.is_leaf:
            current = current.children[current.num_keys]

        last = current.num_keys - 1

        return (current.keys[last], current.elements[last])

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
        """

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
        """

        # * Empty tree Case:
        if self._root is None:
            return

        # * 1 key root case:
        if self._root.num_keys == 1 and self._root.is_leaf and self._root.keys[0] == key:
            self._root = None
            self._total_nodes -= 1
            return

        # * Case 0: Root Case:
        idx = 0
        while idx < self._root.num_keys and key > self._root.keys[idx]:
            idx += 1  # increment counter

        # key match.
        if idx < self._root.num_keys and key == self._root.keys[idx]:
            # * Case 0A: Root is a leaf? delete and complete (no further action)
            if self._root.is_leaf:
                self._root.keys.delete(idx)
                self._root.elements.delete(idx)
                self._total_nodes -= 1
                return
            # * Case 0B: root is internal node & has 1 child after deletion -- (merge nodes)
            elif not self._root.is_leaf and len(self._root.children) == 1:
                # merge root with child - child becomes new root.
                self._root = self._root.children[0]
                self._root.leaf = True
                self._total_nodes -= 1
                return

        return self._recursive_delete(self._root, key)

    def _case_1_leaf_node_contains_key(self, parent_node, idx):
        """ Case 1A: current has min + 1 keys:"""
        if parent_node.num_keys > self.min_keys:
            parent_node.keys.delete(idx)
            parent_node.elements.delete(idx)

    def _case_2_internal_node_contains_key(self, parent_node, idx, key):
        """Case 2A: child i has the min + 1 required keys"""
        # init family unit
        child = parent_node.children[idx]
        left_sibling = parent_node.children[idx-1]
        right_sibling = parent_node.children[idx+1]

        if child.num_keys > self.min_keys:
            # find predecessor:
            pred, pred_idx = self._predecessor(child, idx)
            parent_node.keys[idx] = pred.keys[pred_idx]
            self._recursive_delete(pred, key)

        # * Case 2B: child i has min keys, and its right sibling has min + 1 keys
        if child.num_keys == self.min_keys and right_sibling.num_keys > self.min_keys:
            # find successor:
            succ, succ_idx = self._successor(right_sibling, idx)
            parent_node.keys[idx] = succ.keys[succ_idx]
            self._recursive_delete(succ, key)

        # * Case 2C: both child i and sibling i+1 have min keys. (cant borrow need to merge.)
        if child.num_keys == self.min_keys and (right_sibling.num_keys == self.min_keys or left_sibling.num_keys == self.min_keys):
            # merge right sibling into child
            if right_sibling:
                merged_child = self.merge_right_into_child(parent_node, idx)
                # recursively move into the next child.
                self._recursive_delete(merged_child, key)
            # merge child into left sibling (affects index order)
            else:
                merged_left_sibling = self.merge_with_left(parent_node, idx)
                self._recursive_delete(merged_left_sibling, key)

    def _case_3A_internal_node_does_not_contain_key(self, parent_node, idx, key):
        """
        Child i has min (t-1) keys, but sibling has min + 1 keys -- (borrow from sibling)
        Borrow median key from parent. and swap this with sibling.
        """
        # init family unit
        child = parent_node.children[idx]
        left_sibling = parent_node.children[idx - 1]
        right_sibling = parent_node.children[idx+1]

        # borrow key from left sibling
        if child.num_keys == self.min_keys and left_sibling and left_sibling.num_keys > self.min_keys:
            self.borrow_left(parent_node, idx)
        # borrow key from right sibling
        elif child.num_keys == self.min_keys and right_sibling and right_sibling.num_keys > self.min_keys:
            self.borrow_right(parent_node, idx)
        # recursively move to child and run the same borrow / check
        self._recursive_delete(child, key)

    def _case_3B_internal_node_does_not_contain_key(self, parent_node, idx, key):
        """
        Case 3B:  Child and siblings have min keys (merge child with sibling)
        we need to move a key from current node to become the median key for this new merged node.
        Merging with right is preferable because it maintains index order.
        """
        # init family unit
        child = parent_node.children[idx]
        left_sibling = parent_node.children[idx-1]
        right_sibling = parent_node.children[idx+1]
        # merge with right sibling (this maintains index order)
        if right_sibling:
            merged_child_from_right = self.merge_right_into_child(parent_node, idx)
            self._recursive_delete(merged_child_from_right, key)
        # merge with left sibling (if it exists.)
        elif left_sibling:
            merged_left = self.merge_with_left(parent_node, idx)
            self._recursive_delete(merged_left, key)

    def _recursive_delete(self, node, key) -> None:
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

        # traverse through keys.
        while idx < parent_node.num_keys and key > parent_node.keys[idx]:
            idx += 1  # increment counter

        # init family unit
        child = parent_node.children[idx]
        left_sibling = parent_node.children[idx-1] if idx > 0 else None
        right_sibling = parent_node.children[idx+1] if idx + 1 < parent_node.num_keys + 1 else None

        # * Case 3A:  Child i has min key, but sibling has min + 1 keys -- (borrow from sibling)
        # essentially this works like a pre-emptive check - enforcing that every child has over the min required keys.
        if idx < parent_node.num_keys and key != parent_node.keys[idx] and not parent_node.is_leaf:
            self._case_3A_internal_node_does_not_contain_key(parent_node, idx, key)

        # * Case 3B:  Child and siblings have min keys (merge child with sibling)
        elif not parent_node.is_leaf and left_sibling.num_keys == self.min_keys and right_sibling.num_keys == self.min_keys:
            self._case_3B_internal_node_does_not_contain_key(parent_node, idx, key)

        # * Case 2: Internal node contains the key.
        if idx < parent_node.num_keys and key == parent_node.keys[idx] and not parent_node.is_leaf:
            self._case_2_internal_node_contains_key(parent_node, idx, key)

        # * Case 1: Leaf Node Contains Key: delete immmediately.
        if idx < parent_node.num_keys and key == parent_node.keys[idx] and parent_node.is_leaf:
            self._case_1_leaf_node_contains_key(parent_node, idx)

    # ----- Traversal -----

    def inorder(self):
        """inorder traversal for b trees -- traverses from smallest key to largest key."""
    
    
    def preorder(self):
        """dfs traversal - also called preorder. depth first search"""
    
    def postorder(self):
        """dfs but goes from last to first. not the same as reversing preorder."""

    def levelorder(self):
        """bfs - breadth first search - travels each height level iteratively first, before moving to the next level of the tree"""

    def traverse(self) -> Iterable[Tuple]:
        """
        Traverse throughout the B Tree and return a sequence of all the kv pairs in the tree. 
        In a specifed order (preorder, postorder, levelorder, inorder...)
        """
        return super().traverse()

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
        """
        # child - retains the first half of the keys
        child_node: BTreeNode = parent_node.children[index]
        # * we create a new sibling - it will inherit its leaf status from its other sibling (the child)
        new_sibling: BTreeNode = BTreeNode(self._datatype, self._degree, is_leaf=child_node.leaf)
        self._total_nodes += 1

        # * collect the largest keys and elements from the child. and give them to the sibling.
        # moves the minimum number of keys necessary to the new node.
        for idx in range(self.min_keys):
            # copies over the keys that are higher than the min number of keys.
            new_sibling.keys.append(child_node.keys[idx + self._degree])
            new_sibling.elements.append(child_node.elements[idx+self._degree])
        # copy over children also.
        if not child_node.is_leaf:
            for idx in range(self._degree):
                new_sibling.children.append(child_node.children[idx + self._degree])
        # delete the second half of keys and children from child node.
        for _ in range(self.min_keys):
            child_node.keys.delete(self._degree)
            child_node.elements.delete(self._degree)
        if not child_node.is_leaf:
            for _ in range(self._degree):
                child_node.children.delete(self._degree)

        # * relink parent and new child. (and add promoted key)
        # add new sibling to parent's children list
        parent_node.children.insert(index+1, new_sibling)

        # now insert promoted median key.
        parent_node.keys.insert(index, child_node.keys[self._degree])

    def merge_right_into_child(self, parent_node, idx):
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

        # merge right sibling into child
        child.keys.append_many(right_sibling.keys)
        child.elements.append_many(right_sibling.elements)
        child.children.append_many(right_sibling.children)

        # remove median key / element from parent
        parent_node.keys.delete(idx)
        parent_node.elements.delete(idx)

        # remove right sibling from parent
        parent_node.children.delete(idx + 1)

        return child

    def merge_with_left(self, parent_node, idx) -> BTreeNode[T]:
        """
        Merges a child node into its left sibling. for this it uses its parent's node's median key. (its passed down)
        the child is then removed from the parent...
        """
        child = parent_node.children[idx]
        left_sibling = parent_node.children[idx - 1]

        # move parent key down to become median key for new merged node.
        median_key = parent_node.keys[idx - 1]
        median_element = parent_node.elements[idx - 1]

        # merge left sibling & child
        # append median key to the array.
        left_sibling.keys.append(median_key)
        left_sibling.elements.append(median_element)
        # now append the child keys and elements.
        for i in child.keys: left_sibling.keys.append(i)
        for i in child.elements: left_sibling.elements.append(i)
        for i in child.children: left_sibling.children.append(i)

        # delete key / element from parent.
        parent_node.keys.delete(idx - 1)
        parent_node.elements.delete(idx - 1)
        # remove child from parent.
        parent_node.children.delete(idx)

        return left_sibling

    def borrow_left(self, parent_node, idx) -> None:
        """
        Borrows the last key / element from the left sibling and moves it up to the parent.
        then moves the corresponding parent key / element down to the RIGHT child
        assumes the nodes involved are internal nodes.
        The key separating the two nodes is at index (idx - 1)
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
        child.children.insert(0, left_sibling.children[last])
        left_sibling.children.delete(last)

        # delete key from left sibling
        left_sibling.keys.delete(last)
        left_sibling.elements.delete(last)

    def borrow_right(self, parent_node, idx) -> None:
        """
        Borrows the first key / element from the right sibling and moves it up to the parent.
        Then Moves the Corresponding parent key / element down into the LEFT child.
        parent key --> child key.
        right_sibling key --> parent key
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
        child.children.append(right_sibling.children[0])
        right_sibling.children.delete(0)

        # delete first key from right sibling.
        right_sibling.keys.delete(0)
        right_sibling.elements.delete(0)


# ------------------------------- Main: Client Facing Code: -------------------------------

def main():
    b = BTree(str, 2)
    b.insert(1, "apple")
    b.insert(2, "banana")
    print(repr(b))
    print(b._root)


if __name__ == "__main__":
    main()
