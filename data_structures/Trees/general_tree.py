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


"""
General Tree Implementation: N-ary Tree.
using nodes.
"""

T = TypeVar("T")


class TreeADT(ABC, Generic[T]):
    """Tree"""

    # ----- Canonical ADT Operations -----

    # ----- Accessors -----
    @property
    @abstractmethod
    def root(self) -> T:
        """Returns the value of the Root Node"""
        pass

    @abstractmethod
    def parent(self, node) -> Optional["iNode[T]"]:
        """returns the parent NODE of a specified node"""
        pass

    @abstractmethod
    def child(self, node) -> Optional["iNode[T]"]:
        """returns the child NODE of a specified node"""
        pass

    @abstractmethod
    def num_children(self, node) -> int:
        """returns the total number of children of a specified node"""
        pass

    @abstractmethod
    def is_root(self, node) -> bool:
        """returns true if the node is the root of a tree"""
        pass

    @abstractmethod
    def is_leaf(self, node) -> bool:
        """returns True if the node is a leaf node (no children)"""
        pass

    @abstractmethod
    def is_internal(self, node) -> bool:
        """returns True if the node has children nodes."""
        pass

    @abstractmethod
    def size(self) -> int:
        """returns total number of nodes in the tree"""
        pass

    @abstractmethod
    def depth(self, node) -> int:
        """returns Number of edges from the ROOT down to the specified node"""
        pass

    @abstractmethod
    def height(self, node) -> int:
        """returns Max Number of edges from a specified node to a leaf node (no children)."""
        pass

    # ----- Mutators -----
    @abstractmethod
    def createTree(self, value: T) -> Optional["iNode[T]"]:
        """creates a new tree with a root node"""
        pass

    @abstractmethod
    def addChild(self, parent_node, value) -> "iNode[T]":
        """adds a child node to the specified node."""
        pass
 
    @abstractmethod
    def remove(self, node) -> Optional["iNode[T]"]:
        """removes a specified node and all its descendants"""
        pass

    @abstractmethod
    def replace(self, node, value) -> "iNode[T]":
        """replaces a value in a specified node"""
        pass

    # ----- Traversals -----
    @abstractmethod
    def preorder(self) -> Optional[list[T]]:
        """Depth First Search: (DFS) -- travels from root to last child - returns a list of values"""
        pass

    @abstractmethod
    def postorder(self) -> Optional[list[T]]:
        """Depth First Search: (DFS) travels from last child to root - returns a list of values"""
        pass

    @abstractmethod
    def level_order(self) -> Optional[list[T]]:
        """Breadth First Search: (BFS) --- visiting nodes level by level, - starts from left -> right, and traverses the entire tree top -> bottom"""
        pass

    # ----- Meta Collection ADT Operations -----
    @abstractmethod
    def is_empty(self) -> bool:
        pass

    @abstractmethod
    def __len__(self) -> int:
        """returns total number of nodes in the tree"""
        pass

    @abstractmethod
    def clear(self) -> None:
        pass

    @abstractmethod
    def __contains__(self, value: T) -> bool:
        pass

    @abstractmethod
    def __iter__(self) -> Generator[T, None, None]:
        pass


# Implementation:
class GeneralTree(TreeADT[T]):
    def __init__(
            self, 
            iteration_type: Literal['pre order', 'post order', 'level order']='pre order',
            ) -> None:
        self._root: Optional[iNode[T]] = None
        self.iteration_type = iteration_type

    # ----- Utilities -----

    def view(self):
        """
        Traverses the Tree via stack
        adds connector symbols in front of each node value, depending on whether it is the last child "└─" or one of many "├─",
        every node adds either " " if parent is last child (no vertical bar needed) or "| " if parent is not last child (vertical bar continues)
        the node & its display symbols are appended to a list for the final string output.
        """
        if self.root is None:
            return f"< 🌳 empty tree>"

        hierarchy = []
        tree = [(self.root, "", True)]  # node, prefix, is_last

        while tree:
            # we traverse depth-first, which naturally fits a hierarchical print.
            node, prefix, is_last = tree.pop()

            # root (depth = 0), we print 🌲
            if node is self.root:
                indicator = "🌲:"
            # decides what connector symbol appears before the node value when printing the tree.
            else: 
                indicator = "" if prefix == "" else ("└─" if is_last else "├─")

            # add to final string output
            hierarchy.append(f"{prefix}{indicator}{node.value}") 

            # Build prefix for children - Vertical bars "│" are inherited from ancestors that are not last children
            new_prefix = prefix + ("   " if is_last else "│  ")

            # Iterates over the node’s children in reverse. (left to right) --- enumerate gives index i for calculating new prefix.
            for i, child in enumerate(reversed(node.children)):
                last_child = (i==0)
                # Update ancestor flags: current node's is_last boolean affects all its children
                tree.append((child, new_prefix, last_child))    

        return "\n".join(hierarchy)

    def flattened_view(self):
        # utilizes __iter__ which has 3 different traversal algos
        node_values = [node for node in self]    
        return f"[{', '.join(node_values)}]"

    def bfs_view(self):
        """
        each level should have its own list. so we will have a MD list. then loop through these to sort by levels
        use bfs to traverse tree.
        """

        tree  = [self.root]
        results = []
        infostrings = []

        while tree:
            level = []
            level_size = len(tree)
            for _ in range(level_size):
                node = tree.pop(0)
                level.append(node.value)
                tree.extend(node.children)

            results.append(level)

        for i, level_list in enumerate(results):
            level_string = f"Level: {i}: {', '.join(level_list)}"
            infostrings.append(level_string)

        joined_info = '\n'.join(infostrings)
        final = f"Tree: (BFS) 🌳: Total Nodes: {len(self)}\n{joined_info}"
        return final

    def __str__(self) -> str:
        hierarchy = self.view()
        return hierarchy

    def __repr__(self) -> str:
        class_name = f"<{self.__class__.__qualname__} object at {hex(id(self))}>, Tree Size: {len(self)} Nodes"
        return class_name

    # ----- Canonical ADT Operations -----

    # ----- Accessors -----
    @property
    def root(self):
        """Returns the value of the Root Node"""
        return self._root

    def parent(self, node):
        """returns the parent NODE of a specified node"""
        return node.parent

    def child(self, node):
        """returns a list of all child nodes of a specified node"""
        return node.children

    def num_children(self, node):
        """returns the total number of children of a specified node"""
        return len(node.children)

    def is_root(self, node):
        """returns true if the node is the root of a tree"""
        return node == self._root

    def is_leaf(self, node):
        """returns True if the node is a leaf node (no children)"""
        return len(node.children) == 0

    def is_internal(self, node):
        """returns True if the node has children nodes."""
        return len(node.children) > 0

    def size(self):
        """returns total number of nodes in the tree"""
        return self.current_size

    def depth(self, node):
        """returns Number of edges from the ROOT to the specified node -- Algorithm: traverse up parents until root."""
        depth = 0   # tracks the level from target node
        current_node = node
        while current_node.parent:
            current_node = current_node.parent
            depth += 1
        return depth

    def height(self, node):
        """returns Max Number of edges from a specified node to a leaf node (no children). -- Algorithm: recursively compute the max height of children."""
        if self.is_leaf(node):  # leaf nodes have 0 height
            return 0
        max_height = max(self.height(i) for i in node.children)
        return 1 + max_height   # height must be 1 or over, because not a leaf

    # ----- Mutators -----
    def createTree(self, value):
        """creates a new tree with a root node"""
        self._root = Node(value)
        return self.root

    def addChild(self, parent_node, value):
        """adds a child node to the specified node."""
        child = Node(value)
        child.parent = parent_node  # link to parent.
        parent_node._children.append(child) # link  parent to child.
        return child

    def remove(self, node):
        """
        removes a specified node and all its descendants
        """

        if node is None:  # existence check
            return

        # 1. Store Node & Subtree -- Capture subtree size before modifying anything
        deleted_node = node  # store node to return later

        # 2. Unlink from parent BEFORE deleting parent pointers
        parent = node.parent
        if parent is not None:
            parent.children.remove(node)

        # 3. Iteratively dereference Node & subtree using stack
        subtree = [node]    # note its the actual node input not a variable.
        while subtree:
            node = subtree.pop()
            subtree.extend(node.children)
            node.children = []  # empties list of children
            # dereferences parent node so it no longer points to the node. (becomes a leaf node)
            node.parent = None

        return deleted_node

    def replace(self, node, value):
        """replaces a value in a specified node"""
        replace_node = node
        replace_node.value = value
        return replace_node

    # ----- Traversals -----
    def preorder(self):
        """Depth First Search: (DFS) -- travels from root to last child (top -> bottom) - left -> right returns a list of values"""
        if not self.root:
            return []

        results = []    # stores all node values in a list.
        tree = [self.root]  # acts like a stack - replace with custom stack later.

        while tree:
            node = tree.pop()
            results.append(node.value)
            # adds children to the list so they can also be appended...
            # reversed is needed for the pop to remove the leftmost child first. (to maintain the correct order of the tree.)
            tree.extend(reversed(node.children))    # LIFO

        return results

    def postorder(self):
        """Depth First Search: (DFS) travels from last child to root - returns a list of values"""
        if not self.root:
            return []

        results = [] # stores all node values in a list.
        tree = [self.root]
        temp_storage = []   # results stored in logical order - but popped from the end into the final list.

        # the algo works backwards - from the top - to the bottom, left to right like preorder.
        # the magic comes from popping - by popping teh results from temp storage (the end result). we essentially reverse the order of the nodes.
        while tree: # LIFO
            node = tree.pop()
            temp_storage.append(node)
            tree.extend(node.children)
        # this is the magic
        while temp_storage:   # LIFO
            node = temp_storage.pop()   # grabs the last element
            results.append(node.value)  # adds to the end of the final list.

        return results

    def level_order(self):
        """Breadth First Search: (BFS) -- traverses the tree horizontally a level at a time."""
        if not self.root:
            return []
        results = []
        # this time its operating like a queue. replace with custom queue later.
        tree = [self.root]   

        while tree:
            node = tree.pop(0)  # FIFO -- removes from the front.
            results.append(node.value)
            tree.extend(node.children)  # equivalent to Enqueue (at the rear)

        return results

    # ----- Meta Collection ADT Operations -----
    def is_empty(self):
        return self.root is None

    def __len__(self):
        """returns total number of nodes in the tree"""
        if self.root is None:
            return 0

        tree = [self.root]
        total_nodes = 0

        while tree:
            node = tree.pop()
            total_nodes += 1
            tree.extend(node.children)

        return total_nodes

    def clear(self):
        """The tree is completely emptied and all children nodes are dereferenced."""
        tree = [self.root]
        while tree:
            node = tree.pop()
            tree.extend(node.children)  # add children to the processing line.
            # dereference node
            node.children = []
            node.parent = None

        self.root = None

    def __contains__(self, value):
        """checks to see if any nodes in the tree contain the value."""

        if not self.root:
            return False

        tree = [self.root]  # change to custom stack later....

        while tree:
            node = tree.pop()
            if node.value == value:
                return True
            tree.extend(reversed(node.children))

        return False

    def _preorder_iteration(self):
        """generator version of preorder traversal - goes top to bottom first, then left to right."""
        tree = [self.root]

        while tree:
            node = tree.pop()
            yield node.value
            tree.extend(reversed(node.children))

    def _postorder_iteration(self):
        """generator for postorder traversal - goes bottom to top, left to right."""
        tree = [self.root]
        temp = []

        while tree:
            node = tree.pop()
            temp.append(node)
            tree.extend(node.children)

        while temp:
            node = temp.pop()
            yield node.value

    def _levelorder_iteration(self):
        """generator for BFS - goes level by level, left to right."""        
        tree = [self.root]
        while tree:
            node = tree.pop(0)
            yield node.value
            tree.extend(node.children)

    def __iter__(self):
        """iterates over the tree via 3 traversal methods, DFS, DFS reversed & BFS"""
        if self.iteration_type == 'pre order':
            return self._preorder_iteration()
        elif self.iteration_type == 'post order':
            return self._postorder_iteration()
        elif self.iteration_type == 'level order':
            return self._levelorder_iteration()
        else:
            raise KeyError(f"Error: Iteration Type: {self.iteration_type} is Invalid.")


# Node interface
class iNode(ABC, Generic[T]):
    """interface for Tree ADT node"""
    @property
    @abstractmethod
    def value(self) -> T:
        """return the value stored inside the node"""
        pass
    @property
    @abstractmethod
    def parent(self) -> Optional["iNode[T]"]:
        """return the parent node or None if this is the root"""
        pass
    @property
    @abstractmethod
    def children(self) -> Iterable["iNode[T]"]:
        """return a list of all the children nodes"""
        pass

    # ----- Mutators -----
    @abstractmethod
    def add_child(self, value: T) -> "iNode[T]":
        """insert a child under this node"""
        pass

    @abstractmethod
    def remove_child(self, node: "iNode[T]") -> "iNode[T]":
        """removes a specific child node"""

    # ----- Accessors -----
    @abstractmethod
    def num_children(self) -> int:
        """returns the total number of children of a specified node"""
        pass

    @abstractmethod
    def is_root(self) -> bool:
        """returns true if the node is the root of a tree"""
        pass

    @abstractmethod
    def is_leaf(self) -> bool:
        """returns True if the node is a leaf node (no children)"""
        pass

    @abstractmethod
    def is_internal(self) -> bool:
        """returns True if the node has children nodes."""
        pass


class Node(iNode, Generic[T]):
    """ Node for general tree implementaiton """
    def __init__(self, value: T) -> None:
        self._value = value
        self._parent: Optional[iNode] = None
        self._children: Iterable[iNode] = []

    @property
    def value(self):
        return self._value
    @value.setter
    def value(self, value):
        self._value = value

    @property
    def parent(self):
        return self._parent
    @parent.setter
    def parent(self, value):
        self._parent = value

    @property
    def children(self):
        return self._children
    @children.setter
    def children(self, value):
        self._children = value


    # ----- Utilities -----
    def visualize(self):
        pass

    def __str__(self) -> str:
        """ lists all the values of the node and its children"""
        subtree = [self]
        results = []
        while subtree:
            node = subtree.pop()
            results.append(node.value)
            subtree.extend(reversed(node.children))

        return str(results)

    def __repr__(self) -> str:
        """ Object description """
        class_name = f"<{self.__class__.__qualname__} object at {hex(id(self))}>, Node Data: {self.value}, Direct Children: {self.num_children()}"
        return class_name

    # ----- Mutators -----
    def add_child(self, value):
        """insert a child under this node"""
        new_node = Node(value)
        new_node.parent = self
        self._children.append(new_node)
        return new_node

    def remove_child(self, node):
        """
        removes a specific child node
        Step 1: Store node for return
        Step 2: unlink child - remove from children list
        Step 3: traverse child node subtree - and dereference all nodes
        Step 4: return node.
        """
        if node not in self.children:  # existence check
            raise ValueError(f"Error: Node {node} is not a child of this node.")

        deleted_node = node

        subtree = [node]    # reference subtree in a list(stack)

        # removes node from children list.
        self._children.remove(node)

        # dereference children.
        # By the end of this loop, the entire subtree is disconnected and ready for garbage collection.
        while subtree:
            current_node = subtree.pop()
            subtree.extend(current_node.children)
            # dereference nodes - both children and parent
            current_node.children = []
            current_node.parent = None

        return deleted_node

    # ----- Accessors -----
    def num_children(self):
        """returns the total number of children of a specified node -- ONLY counts direct children."""
        return len(self._children)

    def is_root(self):
        """returns true if the node is the root of a tree"""
        return self._parent is None

    def is_leaf(self):
        """returns True if the node is a leaf node (no children)"""
        return len(self._children) == 0

    def is_internal(self):
        """returns True if the node has children nodes."""
        return len(self._children) > 0


# Main ---- Client Facing Code
def main():

    # -------------- Testing Node Solo Functionality -----------------
    node_a = Node("NODE ROOT")
    print(repr(node_a))
    print(node_a)
    child_a = node_a.add_child("new String to test")
    child_b = node_a.add_child("woatttt are you saying mate?")
    child_bb = child_b.add_child("ill fuck you up....")
    print(node_a)
    print(f"Number of direct children for node_a: {node_a.num_children()}")
    removed = node_a.remove_child(child_a)
    print(node_a)
    print(f"Testing is_root: {child_b.is_root()}")
    print(f"Testing is_leaf: {child_b.is_leaf()}")
    print(f"Testing is_internal: {node_a.is_internal()}")
    print(repr(node_a)) 

    # -------------- Testing Tree Functionality -----------------
    tree = GeneralTree(iteration_type="pre order")
    print(repr(tree))
    print(f"Testing is_empty: {tree.is_empty()}")
    root = tree.createTree("ROOT")
    print(f"Adding Child to Tree:")
    child_a = tree.addChild(root, "a child of summer")
    child_b = tree.addChild(child_a, "a child of winter")
    child_c = tree.addChild(child_a, "a child of spring")
    child_d = tree.addChild(root, "a child of autumn")
    child_dd = tree.addChild(child_d, "fall colors")
    child_de = tree.addChild(child_dd, "fall wind")
    child_e = tree.addChild(root, "E")
    print(f"Testing is_empty: {tree.is_empty()}")
    print(tree)
    print(tree.bfs_view())
    tree.remove(child_de)
    tree.remove(child_c)
    print(tree)

    # test iteration via different traversal algos
    # test depth & hegiht
    # test size
    # test num children, parent, child


if __name__ == "__main__":
    main()
