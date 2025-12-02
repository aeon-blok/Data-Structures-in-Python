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
import math
from collections.abc import Sequence

# endregion


# region custom imports
from user_defined_types.generic_types import T, K, iKey
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

from user_defined_types.tree_types import NodeColor




# endregion

class TreeNodeUtils:
    """Utility Methods for Tree Nodes"""
    def __init__(self, tree_node_obj) -> None:
        self.obj = tree_node_obj

    def validate_tnode(self, node, node_type: type):
        """ensures the specified child belongs to this node."""
        if node is None:
            raise NodeEmptyError("Error: Node is None.")
        if not isinstance(node, node_type):
            raise DsTypeError("Error: Node is not a valid Node Type.")
        if not node.alive:
            raise NodeDeletedError(f"Error: Node has already been deleted.")
        if node not in self.obj.children:  # existence check
            raise NodeOwnershipError(f"Error: Node {node} is not a child of this node.")

    def validate_node_binary_search_key(self, key):
            """ensures the the input key, is a valid key."""
            if not isinstance(key, iKey):
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
        elif not node.alive:
            raise NodeDeletedError("Error: This Node has been deleted and cannot be utilized in tree networks.")
        elif node.tree_owner is not self.obj:
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

    def dfs_depth_first_search(self, target_node, node_type: type):
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

    def reverse_dfs_postorder_search(self, target_node, node_type: type):
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

    def bfs_breadth_first_search(self, target_node, node_type: type):
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
        tree_nodes = CircularArrayDeque(node_type)
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
            # ! Bug: level_node_elements is a stack of nodes, then ', '.join(level_node_elements) will fail
            # ! because ArrayStack does not implement __iter__ returning strings. You likely need:
            # ! ', '.join(str(n) for n in level_node_elements)
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
            if not isinstance(key, iKey):
                raise KeyInvalidError("Error: Input Key is not valid. All keys must be hashable, immutable & comparable (<, >, ==, !=)")
            elif key is None:
                raise KeyInvalidError("Error: Key cannot be None Value")

    def check_key_is_same_type(self, key):
        """Checks the input key type with the stored hash table key type."""
        if self.obj._tree_keytype is None:
            self.obj._tree_keytype = key.datatype
        elif key.datatype != self.obj._tree_keytype:
            raise KeyInvalidError(f"Error: Input Key Type Invalid. Expected: {self.obj._pqueue_keytype.__name__}, Got: {key.datatype.__name__}")

    def bst_descent(self, node, node_type, key):
        """
        descent algorithm - traverses the bst
        node key & key matches? return match
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


    # region AVL Trees

    def avl_tree_max_balance_factor(self, node_type:type):
        """checks all the nodes in the graph to ensure none are above the required balance factor or height property."""

        max_balance = -math.inf
        storage_stack = ArrayStack(int)

        # main case: traverse tree - and count nodes
        tree_nodes = ArrayStack(node_type)
        tree_nodes.push(self.obj.root)
        while tree_nodes:
            current_node = tree_nodes.pop()
            storage_stack.push(current_node.balance_factor)
            # add children to the stack
            if current_node.right is not None:
                tree_nodes.push(current_node.right)
            if current_node.left is not None:
                tree_nodes.push(current_node.left)
        
        while storage_stack:
            balance_factor = storage_stack.pop()
            if balance_factor > max_balance:
                max_balance = balance_factor
        
        return max_balance

    def avl_tree_check_unbalanced(self, node_type:type):
        """checks all the nodes in the graph to ensure none are above the required balance factor or height property."""
        if self.obj.root is None: return False
        # main case: traverse tree - and count nodes
        tree_nodes = ArrayStack(node_type)
        tree_nodes.push(self.obj.root)
        while tree_nodes:
            current_node = tree_nodes.pop()
            if current_node.unbalanced:
                return True
            # add children to the stack
            if current_node.right is not None:
                tree_nodes.push(current_node.right)
            if current_node.left is not None:
                tree_nodes.push(current_node.left)
        return False
    
    def avl_count_tree_nodes(self, node_type: type):
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
    
    def _relink(self, unbalanced_node, child, sibling_subtree):
        """relinks rotated nodes -- not used at the moment due to debug issues"""
        child.parent = unbalanced_node.parent
        unbalanced_node.parent = child
        if sibling_subtree:
            sibling_subtree.parent = unbalanced_node
        
    def avl_rotate_left(self, unbalanced_node):
        """"""
        if unbalanced_node is None or unbalanced_node.right is None:
            raise NodeExistenceError(f"Error: Unbalanced Node doesnt exist")
        
        child = unbalanced_node.right
        sibling_subtree = child.left    # T2 Subtree

        # perform rotation
        child.left = unbalanced_node
        unbalanced_node.right = sibling_subtree

        # relink nodes
        child.parent = unbalanced_node.parent
        unbalanced_node.parent = child
        if sibling_subtree:
            sibling_subtree.parent = unbalanced_node

        # update height
        unbalanced_node.update_height()
        child.update_height()

        return child

    def avl_rotate_right(self, unbalanced_node):
        """rotate right - used with LL and LR rotations"""


        if unbalanced_node is None or unbalanced_node.left is None:
            raise NodeExistenceError(f"Error: Unbalanced Node doesnt exist")
        
        child = unbalanced_node.left
        sibling_subtree = child.right    # T2 Subtree

        # perform rotation
        child.right = unbalanced_node
        unbalanced_node.left = sibling_subtree

        # relink nodes
        child.parent = unbalanced_node.parent
        unbalanced_node.parent = child
        if sibling_subtree:
            sibling_subtree.parent = unbalanced_node

        # update height
        unbalanced_node.update_height()
        child.update_height()

        return child

    def rebalance_avl_tree(self, node):
        """
        Rebalances the AVL tree based on the Balance Factor of the current node.
        There are 4 types of rotations
        what we’re doing in AVL rotations is essentially trinode restructuring.
        in Python, if a function reaches the end without a return, it returns None.
        Never rotate until height is correct.
        """
        balance = node.balance_factor
        
        # Left Heavy Subtree
        if balance > 1:
            if not node.left: raise NodeExistenceError(f"Error: node.left is None")
            # Left Left Rotation:
            if node.left.balance_factor >= 0:
                return self.avl_rotate_right(node)
            # Left Right Rotation
            else:
                # first we rotate the left child left.
                node.left = self.avl_rotate_left(node.left)
                return self.avl_rotate_right(node)

        # Right Heavy Subtree
        if balance < -1:
            if not node.right: raise NodeExistenceError(f"Error: node.right is None")
            # Right Right Rotation:
            if node.right.balance_factor <= 0:
                return self.avl_rotate_left(node)
            # Right Left Rotation
            else:
                node.right = self.avl_rotate_right(node.right)
                return self.avl_rotate_left(node)
        
        # if no balancing required - just return node
        return node

    # endregion

    # region Red Black Trees

    def debug_black_height_paths(self):
        """Counts the number of black nodes along every path in the tree"""
        results = []

        def traverse(node, count, current_path=None):
            current_path = current_path or []
            # * sentinel case:
            if node == self.obj.NIL:
                current_path = list(current_path)
                element = "NIL"
                current_path.append(element)
                count += 1
                packed = (current_path, count)
                results.append(packed)
                return
            
            # * main case:
            new_count = count
            if node.color == NodeColor.BLACK:
                new_count += 1
            current_path.append(node.element)

            # recursively search left and right children.
            traverse(node.left, new_count, current_path)
            traverse(node.right, new_count, current_path)

            current_path.pop()
        traverse(self.obj.root, 0)
        return results

    def black_height_debug_print(self):
        results = self.debug_black_height_paths()
        for i, item in enumerate(results):
            element, black_count = item # unpack tuple
            color_count = self._ansi.color(f"{black_count}", Ansi.RED)
            print(f"Tree Path {i}: Count: [{color_count}] Elements: ({element})") 
    
    def debug_count_real_nodes(self, node_type):
        """counts the total number of nodes in the tree that are NOT sentinels"""
        # * root case
        if self.obj.root == self.obj.NIL:
            return 0
        
        # init stack
        tree = ArrayStack(node_type, 1000)
        tree.push(self.obj.root)

        total_real_nodes = 0

        # print(f"Red Property Validation: Beginning Traversal")
        while tree:
            node = tree.pop()

            if node == self.obj.NIL:
                continue

            total_real_nodes += 1   # increment counter
            # traverse
            left = node.left
            right = node.right
            
            # push children onto stack to iteratively traverse.
            if left != self.obj.NIL:
                tree.push(left)
            if right != self.obj.NIL:
                tree.push(right)
      
        return total_real_nodes
    
    def uncle(self, node):
        """discovers and returns which node is the uncle to the current node."""
        # empty nodes case: - return none   
        parent = node.parent
        grandparent = parent.parent
        return grandparent.right if parent == grandparent.left else grandparent.left

    def left_rotate(self, x) -> None:
        """
        red black tree left rotation - can be either parent or grandparent that is rotated (different to avl tree)
        y is the right child of x
        T2 is the left child of y
        T2 becomes the left child of x
        x becomes the child of y
        y becomes the child of the grandparent (rest of the tree)
        """
        # * NIL Rotation Check Case: During rotations, you must never rotate NIL nodes.
        if x == self.obj.NIL or x.right == self.obj.NIL:
            return
        
        assert x != self.obj.NIL or x.right != self.obj.NIL, "Error: Rotation: Cannot be NIL"

        # initialize pivot -- y is the right child of x.
        y = x.right # y will move up to take x's place

        #  *1.) disconnect T2 from y and parent to x
        # we parent the left child of y (T2) to x
        x.right = y.left
        # if the left child of y is not a sentinel - relink it to x (the rotated node)
        if y.left != self.obj.NIL:
            y.left.parent = x

        # * 2.) Connect y to grandparent
        # relink new subtree root (y) to the rest of the tree (grandparent)
        y.parent = x.parent

        # * 3.) Connect x as the child of y
        # root case: if x is the root, y become the new root.
        if x.parent == self.obj.NIL:
            self.obj.root = y
        # relink: we parent x to y (the new subtree root)
        elif x == x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y
        # relink new subtree root (y) to new child (x): 
        y.left = x
        x.parent = y # relink new child to parent

    def right_rotate(self, y):
        """
        red black tree right rotation. - can be either parent or grandparent that is rotated (different to avl)
        notice the parameters are different. it doesnt matter.
        """
        # * NIL Rotation Check Case: During rotations, you must never rotate NIL nodes.
        if y == self.obj.NIL or y.left == self.obj.NIL:
            return
        
        assert y != self.obj.NIL or y.left != self.obj.NIL, "Error: Rotation: Cannot be NIL"

        # initialize pivot -- x is the left child of y
        x = y.left # x will move up to take y's place

        # * 1.) disconnect T2 from x and parent to y
        # we parent the right child of x (T2) to y
        y.left = x.right
        if x.right != self.obj.NIL:
            x.right.parent = y

        # * 2.) Connect x to grandparent
        # 3. relink new subtree root (x) to the rest of the tree
        x.parent = y.parent

        # * 3.) Connect y as the child of x
        # root case: if y was the root of the tree, x become the new root.
        if y.parent == self.obj.NIL:
            self.obj.root = x
        # relink:  we parent y to x (the new subtree root)
        elif y == y.parent.right:
            y.parent.right = x
        else:
            y.parent.left = x
        # relink new subtree root (x) to new child (y): 
        x.right = y
        y.parent = x # relink new child to parent 

    def debug_insertion_print(self, node):
        """debug string for insert logic."""

        if node.color == NodeColor.RED:
            base_color = self._ansi.color(f"{node.color}", Ansi.RED)
        else:
            base_color = f"{node.color}"

        if node.parent.color == NodeColor.RED:
            parent_color = self._ansi.color(f"{node.parent.color}", Ansi.RED) 
        else:
            parent_color = f"{node.parent.color}"

        if node.left.color == NodeColor.RED:
            left_color = self._ansi.color(f"{node.left.color}", Ansi.RED) 
        else:
            left_color = f"{node.left.color}"

        if node.right.color == NodeColor.RED:
            right_color = self._ansi.color(f"{node.right.color}", Ansi.RED) 
        else:
            right_color = f"{node.right.color}"

        parent = f"Parent: {node.parent.element}[{parent_color}]"
        base = f"{node.element}[{base_color}]"
        left = f"L={node.left.element}[{left_color}]"
        right = f"R={node.right.element}[{right_color}]"

        return f"New Insertion: {base}, {parent}, {left}, {right}"

    def check_red_children_are_black(self, node_type):
        """
        checks to ensure that the children of a red node are black.
        traverses the entire tree from the root and checks: Node and either of its children cannot both be red.
        this will throw an error.
        """
        # * root case
        if self.obj.root == self.obj.NIL:
            return
        # init stack
        tree = ArrayStack(node_type, 1000)
        tree.push(self.obj.root)

        # print(f"Red Property Validation: Beginning Traversal")
        while tree:
            node = tree.pop()
            if node == self.obj.NIL:
                continue
            left = node.left
            right = node.right
                
            node_string = f"node: {node.element}[{node.color}]"
            left_string = f"left: {left.element}[{left.color}]"
            right_string = f"right: {right.element}[{right.color}]"
            
            if node.color == NodeColor.RED and (left.color == NodeColor.RED or right.color == NodeColor.RED):
                raise NodeColorError(f"Error: Cannot have a red node with a red child... {node_string}, {left_string}, {right_string}")
        
            # push children onto stack to iteratively traverse.
            tree.push(left)
            tree.push(right)
 
    def assert_sentinel_black(self):
        """asserts that the sentinel global singleton node is Black. With helpful debug info."""
        sentinel_color = self._ansi.color(f"{self.obj.NIL.color}", Ansi.RED)
        sen_parent = f"{self.obj.NIL.parent.element}[{self.obj.NIL.parent.color}]"
        sen_left = f"{self.obj.NIL.left.element}[{self.obj.NIL.left.color}]"
        sen_right = f"{self.obj.NIL.right.element}[{self.obj.NIL.right.color}]"
        assert self.obj.NIL.color == NodeColor.BLACK, f"Error: Sentinel Node must Always be Black! {self.obj.NIL.element}[{sentinel_color}], parent: {sen_parent} L: {sen_left} R: {sen_right}"

    def assert_red_red_violation(self, node):
        """assert that the node and its parent are red at the start of insert fixup."""
        node_info =f"Node: {node.element}[{node.color}]"
        parent_node_info =f"P: {node.parent.element}[{node.parent.color}]"
        gp_info = f"GP: {node.parent.parent.element}[{node.parent.parent.color}]"
        assert node.color == NodeColor.RED and node.parent.color == NodeColor.RED, f"Error: Red Property: Node and its parent node at the beginning of the repair loop must be Red! {node_info}, {parent_node_info}"
        assert node.parent.parent.color == NodeColor.BLACK, f"Error: grandparent node must Be Black! {gp_info}{parent_node_info}{node_info}"

    def check_node_red_property(self, node):
        """Checks a single node to see if a red property violation has occured."""
        if node == self.obj.NIL:
            return
        if node.color == NodeColor.RED:
            if node.left.color == NodeColor.RED or node.right.color == NodeColor.RED:
                node_info = f"Node={node.element}[{node.color}]"
                left = f"L={node.left.element}[{node.left.color}]"
                right = f"R={node.right.element}[{node.right.color}]"
                raise NodeColorError(f"Error: Node in Deletion Fixup Loop -- Violates red property: {node_info}, {left}, {right}")
            
    def assert_only_1_violation(self, node):
        """assert that only one out of the two possible violations for insertion exist at the start of the fixup loop."""
        parent = node.parent
        red_root_violation = (parent is self.obj.NIL and node.color == NodeColor.RED)
        red_red_violation = (parent is not self.obj.NIL and node.color == NodeColor.RED and parent.color == NodeColor.RED)
        assert red_root_violation ^ red_red_violation, f"Error: Red Property: Only 1 Violation can occur at the same time!"

    def repair_red_property(self, node):
        """
        The only Red-Black invariant that can be violated by insertion is: A red node cannot have a red parent.
        “the while loop maintains the following three-part invariant at the start of each iteration of the loop:” ([Cormen et al., 2022, p. 340]
        “Node ́is red.” ([Cormen et al., 2022, p. 340]
        “If ́parent is the root, then ́parent is black.” ([Cormen et al., 2022, p. 340]
        “If the tree violates any of the red-black properties, then it violates at most one of them” ([Cormen et al., 2022, p. 340]
        “the violation is of either property 2 or property 4, but not both.” ([Cormen et al., 2022, p. 340]
        “If the tree violates property 4, it is because both ́node and ́parent are red.” ([Cormen et al., 2022, p. 340]
        """
        assert node != self.obj.NIL, f"Error: Red Property: node is a NIL sentinel.! {node.element}[{node.color}]"
        # print(f"\nRed Property Repair: (Node: {node.element}[{node.color}])")
        # recursive looping fix while parent exists and is red.
        while node.parent != self.obj.NIL and node.parent.color == NodeColor.RED:
            # print(f"Begin Loop")
            # * invariants
            self.assert_only_1_violation(node)
            self.assert_red_red_violation(node)
            self.assert_sentinel_black()

            # initialize node family...
            grandparent = node.parent.parent

            # * root case: grandparent is NIL 
            if grandparent == self.obj.NIL: 
                # print(f"grandparent is NIL Sentinel - exiting loop early.")
                break   # stop recolor

            parent = node.parent
            uncle = self.uncle(node)

            # * ----- Case 1: Uncle and Parent is red (Red Red violation)-----
            # No rotation is needed here because the tree structure is already a valid BST.
            # recolor both parent & uncle to black
            # recolor grandparent to red
            # bubble up through tree - and repeat process if any red red violations occur
            if uncle.is_red:
                # print(f"Case 1: Uncle and Parent is red (Red Red violation)")
                # print(f"Uncle is red: {uncle.element}[{uncle.color}]")
                # Fixes the immediate red-red violation between node and parent.
                parent.set_black()
                assert parent.is_black, "Red Property: Case 1: Parent Should change to Black!"
                # Fixes the red-red violation between the parent and uncle, 
                # indirectly ensuring the grandparent’s subtree remains valid.
                uncle.set_black()
                assert uncle.is_black, "Red Property: Case 1: Uncle Should change to Black!"

                # Recolors the grandparent from black → red. 
                # Now the red-red problem moves up the tree, because grandparent might now violate the property with its own parent.
                grandparent.set_red()
                assert grandparent.is_red, "Red Property: Case 1: Grandparent Should change to RED!"
                # traverses upwards to check for a new red red violations - the loop bubbles up fixing violations as it goes.
                # print(f"traversing up tree: Node Begins as: {node.element}[{node.color}]")
                node = grandparent
                # print(f"traversing up tree: Changes To {node.element}[{node.color}]")
                continue # Restart the while loop with the new 'node' pointer

            # * ----- Case 2: Inner child → rotate parent (double rotation - case 3 is the second part of case 2)-----
            # perform rotation to transform an inner child (zig zag) to an outer child (straignt line)
            if (parent.is_left_child and node.is_right_child) or (parent.is_right_child and node.is_left_child):
                # print(f"Case 2: red property: Inner child → rotate parent (double rotate - case 3 comes directly after)")
                # transform inner child into outer child
                node = parent
                if parent.is_left_child:
                    assert parent.is_left_child, "Red Property: Case 2: parent needs to be left child"
                    self.left_rotate(node)
                else:
                    self.right_rotate(node)

                # reset family trackers.
                # print(f"updating parent pointers:  Parent starts: {parent.element}[{parent.color}]")
                parent = node.parent
                # print(f"updating parent pointers: Becomes: {parent.element}[{parent.color}]")
                # print(f"updating Grandparent pointers: Starts: {grandparent.element}[{grandparent.color}]")
                grandparent = parent.parent
                # print(f"updating Grandparent pointers: Becomes: {grandparent.element}[{grandparent.color}]")
                continue  # <--- THIS IS KEY: Re-enter the loop to handle the now-straight line case

            # ----- Case 3: Outer child → rotate grandparent -----
            # Fixes the red-red violation between node and parent.
            # similar to Case 1: with red uncle - set parent to black. (but not uncle), then set grandparent to red.
            # then rotate on the grandparent to restore balance
            # print(f"Case 3: red property: Outer child → rotate grandparent (left left or right right rotate)")
            parent.set_black()
            # Preserves black-height property (number of black nodes along paths).
            # Any previous violations with grandparent are now pushed up, but we handle that in the loop.
            grandparent.set_red()
            # Pull the red parent up and push the black grand down 
            # restores the tree’s balanced structure.
            if parent.is_left_child:
                self.right_rotate(grandparent)
            else:
                self.left_rotate(grandparent)
                
        # print(f"End loop")
        # Red Black Invariant - must always be true.
        self.obj.root.set_black()
        # print(f"Set root to black: {self.obj.root.element}[{self.obj.root.color}]")
        if node.parent == self.obj.root:
            assert node.parent.color == NodeColor.BLACK, f"Error: If the Parent is the Root, the root must be black! Info: {node.parent.element}[{node.parent.color}]"
        # print(f"Exit Red Property Repair")

    def transplant(self, old_subtree, new_subtree):
        """Used during red black deletion - replaces 1 subtree with another."""
        # * root case:
        if old_subtree.parent == self.obj.NIL:
            self.obj.root = new_subtree
        # * relink the old subtree as the child of the new subtree
        elif old_subtree == old_subtree.parent.left:
            old_subtree.parent.left = new_subtree
        else:
            old_subtree.parent.right = new_subtree
        # * relink new subtree to the rest of the tree.
        new_subtree.parent = old_subtree.parent

    def sibling(self, node):
        """Used during deletion with Red Black Trees. Identifies the sibling of the target node"""
        return node.parent.right if node.is_left_child else node.parent.left

    def repair_black_property(self, replacement_node):
        """
        used when deleting nodes - solves any potential black height property violations
        Black Height Property: Every path from a node to its descendant null nodes (leaves) must have the same number of black nodes.
        Adds checks like if node != NIL before recoloring or propagation.
        the replacement node is considered doubly black. - this will be removed by recoloring and propagating to the root.
        “The key idea is that in each case, the transformation applied preserves the number of black nodes (including x’s extra black)” ([Cormen et al., 2022, p. 350]
        """
        print(f"\nStarting Repair Black Property:")
        while replacement_node != self.obj.root and replacement_node.color == NodeColor.BLACK:
            print(f"Begininning Repair Loop:")
            print(f"replacement={replacement_node.element}[{replacement_node.color}], P: {replacement_node.parent.element} L: {replacement_node.left.element} R: {replacement_node.right.element}")
            sibling = self.sibling(replacement_node) # as per CLRS -- its important to ensure that sibling is not NIL
            print(f"Sibling={sibling.element}[{sibling.color}], P: {sibling.parent.element} L: {sibling.left.element} R: {sibling.right.element}")
            
            # * Case 1: sibling is red
            # recolor sibling to Black, recolor parent to Red, then rotate parent. Update sibling after.
            if sibling != self.obj.NIL and sibling.is_red:
                assert sibling != self.obj.NIL, "Error: Black Property: Sibling cannot be a NIL sentinel."
                assert sibling.color == NodeColor.RED, "Case 1: Error: Sibling Must be red!"
                if sibling != self.obj.NIL:
                    sibling.color = NodeColor.BLACK
                    assert sibling.color == NodeColor.BLACK, "Case 1: Error: Sibling Must be changed to BLACK!"
                # root case: parent is the root
                if replacement_node.parent != self.obj.NIL: 
                    replacement_node.parent.color = NodeColor.RED
                    assert replacement_node.parent.color == NodeColor.RED, "Case 1: Error: Parent Must be changed to RED!"

                    # rotate parent -- when rotating both the rotated node and its swap child cannot be a sentinel.
                    if replacement_node.is_left_child:
                        self.left_rotate(replacement_node.parent)
                    else:
                        self.right_rotate(replacement_node.parent)
                    # update sibling pointer after rotation
                    sibling = self.sibling(replacement_node)

                print(f"Case 1: sibling is red")
                print(f"replacement={replacement_node.element}[{replacement_node.color}]")
                print(f"sibling={sibling.element}[{sibling.color}]")
                print(f"Grandparent Color: {replacement_node.parent.color}")

            # * Case 2: Sibling is black and Both Children are Black
            # 
            if sibling == self.obj.NIL or (sibling.is_black and sibling.left.is_black and sibling.right.is_black):
                print(f"Case 2: Both Siblings Children are Black")
                if sibling != self.obj.NIL:
                    sibling.color = NodeColor.RED
                # propagate upwards...
                if replacement_node.parent != self.obj.root:
                    replacement_node = replacement_node.parent
                    sibling = self.sibling(replacement_node)
                    print(f"replacement={replacement_node.element}[{replacement_node.color}]")
                    print(f"sibling children= L={sibling.left.element}[{sibling.left.color}], R={sibling.right.element}[{sibling.right.color}]")
                    print(f"Grandparent Color: {replacement_node.parent.color}")
                    continue
                else:
                    break

            # * Case 3: Sibling is black and has a near red child 
            # case 3 should not violate any red black properties when it performs the rotation.
            if sibling != self.obj.NIL and sibling.is_black and sibling.right.is_black and sibling.left.is_red:
                # left child is red (the near child is sibling.left)
                if sibling.left != self.obj.NIL and replacement_node.is_left_child:
                    sibling.left.color = NodeColor.BLACK
                    sibling.color = NodeColor.RED
                    self.right_rotate(sibling)
                    sibling = self.sibling(replacement_node)

                    print(f"Case 3: left red near child")
                    print(f"replacement={replacement_node.element}[{replacement_node.color}]")
                    print(f"sibling={sibling.element}[{sibling.color}]")

                # right child is red (mirrors left)
                elif sibling != self.obj.NIL and sibling.is_black and sibling.left.is_black and sibling.right.is_red:
                    if sibling.right != self.obj.NIL and replacement_node.is_right_child: 
                        sibling.right.color = NodeColor.BLACK
                        sibling.color = NodeColor.RED
                        self.left_rotate(sibling)
                        sibling = self.sibling(replacement_node)

                    print(f"Case 3: right red near child")
                    print(f"replacement={replacement_node.element}[{replacement_node.color}]")
                    print(f"sibling={sibling.element}[{sibling.color}]")
                    
            # * Case 4: Sibling is black and has a far red child
            if sibling != self.obj.NIL:
                if replacement_node.parent != self.obj.NIL:
                    sibling.color = replacement_node.parent.color
                    replacement_node.parent.color = NodeColor.BLACK
                # fix far child color (for left - this will be sibling.right)
                if replacement_node.is_left_child and replacement_node.parent != self.obj.NIL:
                    if sibling.right != self.obj.NIL:
                        sibling.right.color = NodeColor.BLACK
                    self.left_rotate(replacement_node.parent)

                    print(f"Case 4: left red outer child")
                    print(f"replacement={replacement_node.element}[{replacement_node.color}]")
                    print(f"sibling={sibling.element}[{sibling.color}]")

                else:
                    if sibling.left != self.obj.NIL:
                        sibling.left.color = NodeColor.BLACK
                        self.right_rotate(replacement_node.parent)

                    print(f"Case 4: right red outer child")
                    print(f"replacement={replacement_node.element}[{replacement_node.color}]")
                    print(f"sibling={sibling.element}[{sibling.color}]")

                replacement_node = self.obj.root
            


        # * finally the replacement node must always be Black:
        if replacement_node != self.obj.NIL:
            replacement_node.color = NodeColor.BLACK

        # * fixup invariants -- must be true after every loop
        assert self.obj.root.color == NodeColor.BLACK, "Error: Black Property: root must always be black at end of fixup loop."
        self.check_node_red_property(replacement_node)
        assert self.obj.is_black_property == True, "Error: Black Property: every path in a tree, must have the same number of black nodes"
        print(f"Exiting Loop:")
        print(f"replacement={replacement_node.element}[{replacement_node.color}], P: {replacement_node.parent.element} L: {replacement_node.left.element} R: {replacement_node.right.element}")
        
    def red_black_descent(self, node, node_type, key):
        """red black search - uses bst descent - but with sentinels rather than None."""
        self.validate_datatype(node_type)
        self.validate_binary_search_key(key)
        self.validate_tree_node(node, node_type)
        while node != self.obj.NIL:
            # match found case:
            if key == node.key: return node
            # key < node key case:
            elif key < node.key: node = self.obj.left(node)
            # key > node key case:
            else: node = self.obj.right(node)
        return None

    def red_black_tree_height(self, edge_based: bool = True):
        """returns max height for binary tree..."""
        if self.obj.root is self.obj.NIL:
            return 0
        start_depth = 0 if edge_based else 1
        tree_nodes = ArrayStack(tuple)  # note the type is a tuple.
        tree_nodes.push((self.obj.root, start_depth))
        max_height_counter = 0

        while tree_nodes:
            current_node, depth = tree_nodes.pop()
            max_height_counter = max(max_height_counter, depth)

            # add children to the stack
            if current_node.right != self.obj.NIL:
                tree_nodes.push((current_node.right, depth + 1))
            if current_node.left != self.obj.NIL:
                tree_nodes.push((current_node.left, depth + 1))

        return max_height_counter

    def red_black_count_tree_nodes(self, node_type: type):
        """binary tree variant for counting the nodes in a tree"""
        self.validate_datatype(node_type)

        # empty case:
        if self.obj.root is self.obj.NIL:
            return 0

        # main case: traverse tree - and count nodes
        tree_nodes = ArrayStack(node_type)
        tree_nodes.push(self.obj.root)
        total_nodes = 0
        while tree_nodes:
            current_node = tree_nodes.pop()
            total_nodes += 1
            # add children to the stack
            if current_node.right is not self.obj.NIL:
                tree_nodes.push(current_node.right)
            if current_node.left is not self.obj.NIL:
                tree_nodes.push(current_node.left)
        return total_nodes

    def red_black_is_sentinel(self, node):
        """error checking to see if the node is a sentinel or not"""
        if node == self.obj.NIL:
            raise NodeEmptyError(f"Error: This Node is a sentinel - it doesnt exist. (all sentinels return black color in a red black tree)")

    def red_black_dfs_traversal(self, target_node, node_type:type):
        """depth first search for binary trees"""

        # empty case:
        if target_node is self.obj.NIL:
            return

        # main case
        tree_nodes = ArrayStack(node_type)
        tree_nodes.push(target_node)

        # traverse tree - add children in reverse order to the stack.
        while tree_nodes:
            current_node = tree_nodes.pop()
            yield current_node
            # NOTICE THE ORDER - its right to left - when pushing to the stack with dfs
            if current_node.right != self.obj.NIL:
                tree_nodes.push(current_node.right)
            if current_node.left != self.obj.NIL:
                tree_nodes.push(current_node.left)  # push to main stack

    def red_black_postorder_traversal(self, target_node, node_type:type):
        """reversed dfs for binary trees"""
        self.validate_datatype(node_type)
        self.validate_tree_node(target_node, node_type)

        if target_node is self.obj.NIL:
            return

        # main case
        tree_nodes = ArrayStack(node_type)
        tree_nodes.push(target_node)
        reverse_stack = ArrayStack(node_type)

        while tree_nodes:
            current_node = tree_nodes.pop()
            reverse_stack.push(current_node)
            # NOTICE: the order is reversed for postorder.
            if current_node.left != self.obj.NIL:
                tree_nodes.push(current_node.left)
            if current_node.right != self.obj.NIL:
                tree_nodes.push(current_node.right)

        while reverse_stack:
            node = reverse_stack.pop()
            yield node

    def red_black_bfs_traversal(self, target_node, node_type: type):
        """breadth first search for binary trees"""
        self.validate_datatype(node_type)
        self.validate_tree_node(target_node, node_type)

        if target_node is self.obj.NIL:
            return

        tree_nodes = CircularArrayDeque(node_type)
        tree_nodes.add_rear(target_node)

        while tree_nodes:
            current_node = tree_nodes.remove_front()
            yield current_node
            if current_node.left != self.obj.NIL:
                tree_nodes.add_rear(current_node.left) 
            if current_node.right != self.obj.NIL:
                tree_nodes.add_rear(current_node.right)

    def red_black_inorder_traversal(self, target_node, node_type: type):
        """
        Inorder traversal for binary trees. Visit nodes in the order Left → Root → Right.
        For a general binary tree, the values won’t be sorted (only for BST)
        """

        if target_node is self.obj.NIL:
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



    # region Disjoint Set














            










    # endregion
