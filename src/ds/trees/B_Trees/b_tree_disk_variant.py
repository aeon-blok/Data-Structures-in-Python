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
import struct
from pathlib import Path

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
from utils.representations import BTreeNodeRepr, BTreeRepr, PageRepr
from utils.helpers import RandomClass
from utils.exceptions import *
from utils.constants import PAGE_SIZE

from adts.collection_adt import CollectionADT
from adts.b_tree_adt import BTreeADT

from ds.sequences.Stacks.array_stack import ArrayStack
from ds.primitives.arrays.dynamic_array import VectorArray, VectorView
from ds.maps.hash_table_with_chaining import ChainHashTable
from ds.trees.tree_nodes import BTreeNode
from ds.trees.tree_utils import TreeUtils

from user_defined_types.generic_types import Index, ValidDatatype, ValidIndex, TypeSafeElement, PositiveNumber
from user_defined_types.key_types import iKey, Key
from user_defined_types.tree_types import NodeColor, Traversal, PageID

# endregion

class Page:
    """
    Used with disk B-Tree - is a fixed size (usually 4096 bytes)
    A Page is a fixed-size in-memory container for a block of bytes.
    Represent one fixed-size block of bytes that corresponds 1-to-1 with a block on disk.
    Page's are identified by an ID number not pointers.
    It allows us to modify the bytes that we can then store to disk storage.
    """

    # Matches common OS page size, Aligns with filesystem block sizes, Minimizes partial reads/writes
    SIZE = PAGE_SIZE

    def __init__(self, page_id: PageID, data: bytes) -> None:
        self.page_id = page_id
        # when we load a page from the disk storage, we pipe it in via the data parameter.
        # A bytearray in Python is a mutable sequence of bytes (integers from 0–255).
        self.data = bytearray(data) if data is not None else bytearray(self.SIZE)
        self._used_bytes: int = len(data) if data is not None else 0

        # composed objects
        self._desc = PageRepr(self)

    @property
    def datatype(self) -> type:
        return bytes

    @property
    def available_space(self) -> int:
        """Returns the number of unused bytes remaining in this page..."""
        return self.SIZE - self._used_bytes

    def get_bytes(self) -> bytes:
        """Return a copy of the in-memory page bytes."""        
        return bytes(self.data)

    def modify_bytes(self, data: bytes) -> None:
        """Replace (inplace) the in-memory page bytes with new data."""

        #  overflow check
        if len(data) != self.SIZE:
            raise DsInputValueError(f"Error: Bytes input exceeds the Page Capacity: {self.SIZE}")

        self.data[:] = data

    def __str__(self) -> str:
        return self._desc.str_page()

    def __repr__(self) -> str:
        return self._desc.repr_page()


class PageManager:
    """
    Interface for writing nodes to disk, and reading nodes from disk.
    PageManager orchestrates serialization, disk writes, and tree structure.
    """

    # todo free page list - used for reusing deleted page ids.

    def __init__(self, location: str, datatype: type, keytype: Optional[type], degree: int) -> None:
        self._auto_id: PageID = 0    
        self.page_table = ChainHashTable(BTreeNode)
        self.pagefile = Path(location)

        if not self.pagefile.exists() or self.pagefile.stat().st_size == 0:
            self.pagefile.touch()
            self._datatype = ValidDatatype(datatype)
            self._keytype = keytype
            self._degree = degree
            self._root_page_id = None
            self.free_list_head = None
            self.free_list_cache = []
            self.write_tree_metadata(self.root_page_id)
        else:
            root_pid, deg, _, dtype, ktype = self.read_tree_metadata()
            self._datatype = ValidDatatype(dtype)
            self._keytype = ValidDatatype(ktype) if ktype is not None else None
            self._degree = deg if deg else degree
            self._root_page_id = root_pid if root_pid else None
            self.free_list_head: Optional[PageID] = None  # on disk implicit linked list
            self.free_list_cache: list[PageID] = []   # in memory
            self.load_free_list_cache() # loads the cache on init.

    @property
    def keytype(self):
        return self._keytype

    @property
    def root_page_id(self) -> Optional[PageID]:
        return self._root_page_id

    @root_page_id.setter
    def root_page_id(self, value: PageID) -> None:
        self._root_page_id = value
        self.write_tree_metadata(value)

    # Free List Cache
    def _read_page_bypass(self, page_id):
        """bypasses the free list check - its used to build a free list in memory cache for quick retrieval"""
        with open(self.pagefile, "rb") as file:
            file.seek(page_id * PAGE_SIZE)
            return file.read(PAGE_SIZE)

    def load_free_list_cache(self):
        """Creates an in memory cache from the stored on disk linked list """

        current = self.free_list_head
        self.free_list_cache = []
        while current:
            self.free_list_cache.append(current)
            page_data = self._read_page_bypass(current)
            # moves to the next free page item in the pagefile metadata section (page 0)
            next_free_page = int.from_bytes(page_data[:4], 'big')
            current = next_free_page if next_free_page != 0 else None

    # private helper methods:

    def _allocate_page_id_via_free_list(self) -> PageID:
        """
        Checks if a "free" page is available (previously deleted page)
        allocates this free slot to a new page.
        returns page id and increments counter
        """
        if self.free_list_cache:
            # return and remove the first element from the free list cache
            page_id = self.free_list_cache.pop(0)
            # updates on-disk free list head for persistent storage of the free list.
            page_bytes = self._read_page_bypass(page_id)
            next_free = int.from_bytes(page_bytes[:4], 'big')
            self.free_list_head = next_free if next_free != 0 else None
            self.write_tree_metadata(self.root_page_id) 
            return page_id
        # no cache? check if on disk free list exists?
        elif self.free_list_head is not None:
            page_id = self.free_list_head
            page_bytes = self._read_page_bypass(page_id)
            next_free = int.from_bytes(page_bytes[:4], 'big')
            self.free_list_head = next_free if next_free != 0 else None
            self.write_tree_metadata(self.root_page_id)  
            return page_id
        # allocate a new page
        else:
            pid = self._allocate_page_id()
            return pid

    def _allocate_page_id(self) -> PageID:
        """
        returns page id and increments counter
        """
        pid = self._auto_id
        self._auto_id += 1
        return pid

    def free_page_id(self, page_id: PageID) -> None:
        """
        Frees up a page slot in the pagefile.
        Adds it to the free list so that the next time a page is stored, it will utilize this slot rather than create a new page.
        Updates both the Free list cache and the free list on disk.
        """

        # Every freed page stores a pointer to the next free page in its first 4 bytes. x00 0 bytes indicates the end of the free list.
        # This is how the freed page “links” to the next free page, forming a persistent on-disk linked list.
        free_list_head_bytes = self.free_list_head.to_bytes(4, "big") if self.free_list_head is not None else b"\x00\x00\x00\x00"

        # load page - it still contains the old node data, we will overwrite it to point to the head of the free list.
        page = self._load_page(page_id)
        page_bytes = bytearray(page.get_bytes())    # conv to mutable bytearray
        # point to the head of the free list
        page_bytes[0:4] = free_list_head_bytes
        page.modify_bytes(bytes(page_bytes))    # apply modification
        self._store_page(page)  # save page

        # update cache and linked list
        # Insert the newly freed page at the front of the cache so the next allocation can reuse it quickly.
        self.free_list_cache.insert(0, page_id)
        # self.free_list_head is updated to the newly freed page’s ID, making it the new head of the on-disk linked free list.
        self.free_list_head = page_id

        # update metadata on disk
        self.write_tree_metadata(self.root_page_id)

    def _encode_node(self, node: BTreeNode):
        """
        Converts a Node into a fixed size byte representation. 
        and adds a page id and children page ids to the bytes.
        """

        # todo overflow handling of cursor (over page size...)

        # * validate node input.

        # * init vars
        buffer = bytearray(PAGE_SIZE)
        # represents the current index in the byte array where the next write should happen.
        cursor: int = 0

        # * start encoding
        # Converts node.is_leaf boolean into int, 1=leaf, 0=internal, We store this as 1 byte in the header
        is_leaf = 1 if node.is_leaf else 0
        # used so we only store actual keys and not empty array slots as bytes.
        num_keys = node.num_keys

        # * start building the struct
        # is leaf boolean
        struct.pack_into("B", buffer, cursor, is_leaf)
        cursor += 1
        # num_keys integer
        struct.pack_into("I", buffer, cursor, num_keys)
        cursor += 4

        # Problem: what if the serialized keys + values exceed PAGE_SIZE?
        # In professional systems (Postgres, SQLite):
        # They either limit the number of keys per page dynamically to fit the page size.
        # Or spill overflow items to a separate page.

        # keys
        for key in range(num_keys):
            key = node.keys[key]
            # serializes key object into bytes
            key_bytes = pickle.dumps(key)
            key_len = len(key_bytes)    # length of bytes
            # "H" → unsigned short (2 bytes) → max value 65535
            struct.pack_into("H", buffer, cursor, key_len)
            cursor +=2
            # copies the key bytes into the buffer
            buffer[cursor:cursor+key_len] = key_bytes
            cursor += key_len   # increment cursor

        # elements
        for e in range(num_keys):
            element = node.elements[e]
            elem_bytes = pickle.dumps(element)
            elem_len = len(elem_bytes)
            struct.pack_into("H", buffer, cursor, elem_len)
            cursor += 2
            buffer[cursor:cursor+elem_len] = elem_bytes
            cursor += elem_len

        # children nodes (leaves dont have children so nothing to store...)
        if not node.is_leaf:
            for child in node.children:

                # existence check
                if child not in self.page_table:
                    raise NodeExistenceError(f"Error: Child Node has not been written yet")

                # retrieve page id for child from page table.
                child_page_id = self.page_table[child]

                # packs the child page id into the buffer as an unsigned int.
                struct.pack_into("I", buffer, cursor, child_page_id)
                cursor += 4

        return bytes(buffer)

    def _decode_node(self, page_bytes: bytes) -> BTreeNode:
        """
        Decodes bytes into a B Tree Node. 
        Must mirror Encode Node exactly
        """

        cursor = 0

        # header
        is_leaf = struct.unpack_from("B", page_bytes, cursor)[0]
        cursor += 1

        num_keys = struct.unpack_from("I", page_bytes, cursor)[0]
        cursor += 4

        # create node object.
        node = BTreeNode(self._datatype, self._degree, is_leaf=bool(is_leaf))
        node.keytype = self._keytype

        # keys
        for i in range(num_keys):
            key_bytes_length = struct.unpack_from("H", page_bytes, cursor)[0]
            cursor += 2

            key_bytes = page_bytes[cursor:cursor+key_bytes_length]
            cursor += key_bytes_length

            key = pickle.loads(key_bytes)
            node.keys.append(key)

        # elements
        for i in range(num_keys):
            elem_bytes_length = struct.unpack_from("H", page_bytes, cursor)[0]
            cursor += 2

            elem_bytes = page_bytes[cursor:cursor+elem_bytes_length]
            cursor += elem_bytes_length

            element = pickle.loads(elem_bytes)
            node.elements.append(element)

        # children
        if not node.is_leaf:
            for i in range(num_keys+1):
                child_page_id = struct.unpack_from("I", page_bytes, cursor)[0]
                cursor += 4
                node.children.append(child_page_id)

        return node

    def _store_page(self, page: Page) -> None:
        """Opens the Pagefile and writes the page object into it."""
        if page.page_id in self.free_list_cache:
            raise NodeDeletedError(f"Error: Page {page.page_id}: Has been deleted and cannot be accessed")
        with self.pagefile.open("r+b") as file:
            file.seek(page.page_id * PAGE_SIZE)
            file.write(page.get_bytes())

    def _load_page(self, page_id: PageID) -> Page:
        """Opens the pagefile and retrieves the specific page (via id) as a Page Object"""
        if page_id in self.free_list_cache:
            raise NodeDeletedError(f"Error: Page {page_id}: Has been deleted and cannot be accessed")
        with self.pagefile.open("rb") as file:
            file.seek(page_id * PAGE_SIZE)
            data = file.read(PAGE_SIZE)
        return Page(page_id, data)

    # Public API Inteface:
    def write_node_to_disk(self, node: BTreeNode) -> PageID:
        """
        writes a B-tree Node to disk, returns the page_id
        recursion is used to write the children first so that the page ids for the children node are correct.
        """

        # * write children recursively first.
        if not node.is_leaf:
            for child in node.children:
                if child not in self.page_table:
                    self.write_node_to_disk(child)

        # * allocate page_id
        page_id: PageID = self._allocate_page_id_via_free_list()
        # * map to page table
        self.page_table.put(node, page_id)

        # * encode node to bytes (with page id and children page id's)
        page_bytes = self._encode_node(node)
        page = Page(page_id, page_bytes)  # wrap in Page Object

        # * write page to disk
        self._store_page(page)

        return page_id

    def read_node_from_disk(self, page_id: PageID) -> BTreeNode:
        """reads bytes from disk, decodes the bytes into a node object. And recursively loads child node objects"""

        # load page bytes
        page = self._load_page(page_id)
        page_bytes = page.get_bytes()

        # decode the page bytes into a node. (this will create a new node with the same data properties as the original node had.)
        node: BTreeNode = self._decode_node(page_bytes)

        # recursively resolve children if internal node
        if not node.is_leaf:
            resolved_children = VectorArray(node.num_keys+1, BTreeNode)
            for page_id in node.children:
                child: BTreeNode = self.read_node_from_disk(page_id)
                resolved_children.append(child)
            # update children array to be the new array with correct node objects instead of Page IDs
            node.children = resolved_children

        return node

    def write_tree_metadata(self, root_page_id: PageID) -> None:
        """
        Writes some simple metadata about the tree, including the root page id. 
        Which is essential for loading a tree from disk.
        """
        buffer = bytearray(PAGE_SIZE)
        cursor = 0

        # root page id
        struct.pack_into("I", buffer, cursor, root_page_id)
        cursor += 4

        # free list head
        free_list_head = self.free_list_head if self.free_list_head else 0
        struct.pack_into("I", buffer, cursor, free_list_head)
        cursor += 4

        # degree
        struct.pack_into("I", buffer, cursor, self._degree)
        cursor += 4

        # number of nodes
        struct.pack_into("I", buffer, cursor, len(self.page_table))
        cursor += 4

        # datatype
        datatype_bytes = pickle.dumps(self._datatype)
        struct.pack_into("H", buffer, cursor, len(datatype_bytes))
        cursor += 2
        buffer[cursor:cursor+len(datatype_bytes)] = datatype_bytes
        cursor += len(datatype_bytes)

        # keytype
        keytype_bytes = pickle.dumps(self._keytype)
        struct.pack_into("H", buffer, cursor, len(keytype_bytes))
        cursor += 2
        buffer[cursor:cursor+len(keytype_bytes)] = keytype_bytes
        cursor += len(keytype_bytes)

        # create page object and write to pagefile.
        page = Page(0, bytes(buffer))
        self._store_page(page)

    def read_tree_metadata(self) -> tuple:
        """
        reads the metadata from the first page in the pagefile.
        this contains the root page id inside a tuple with some extra information
        root_page_id in metadata always points to the current root.
        the degree (specifies the min(deg-1) and max(2deg-1) keys for a node)
        total number of nodes
        the node element datatype
        the key datatype
        """

        page = self._load_page(0)
        buffer = page.get_bytes()
        cursor = 0

        root_page_id = struct.unpack_from("I", buffer, cursor)[0]
        cursor += 4

        free_list_head = struct.unpack_from("I", buffer, cursor)[0]
        self.free_list_head = free_list_head if free_list_head != 0 else None
        cursor += 4

        self._degree = struct.unpack_from("I", buffer, cursor)[0]
        cursor += 4

        num_nodes = struct.unpack_from("I", buffer, cursor)[0]
        cursor += 4

        datatype_bytes_length = struct.unpack_from("H", buffer, cursor)[0]
        cursor += 2
        self._datatype = pickle.loads(buffer[cursor:cursor+datatype_bytes_length])
        cursor += datatype_bytes_length

        keytype_bytes_length = struct.unpack_from("H", buffer, cursor)[0]
        cursor += 2
        self._keytype = pickle.loads(buffer[cursor:cursor+keytype_bytes_length])
        cursor += keytype_bytes_length

        return (root_page_id, self._degree, num_nodes, self._datatype, self._keytype)

    def load_tree_from_disk(self) -> BTreeNode:
        """
        Loads the entire B-Tree from disk
        reads the metadata,
        recursively reads the children of the root and loads them into memory.
        returns the root node.
        """

        # metadata
        root_page_id, degree, num_nodes, datatype, keytype = self.read_tree_metadata()
        self._degree: int = degree
        self._datatype: type = datatype
        self._keytype: type = keytype

        # recursively read children of root to load tree into memory.
        root = self.read_node_from_disk(root_page_id)

        return root

    def save_tree_to_disk(self, root: BTreeNode) -> None:
        """
        Saves te entire B-Tree to Disk
        recursively writes all nodes starting from the root
        updates the tree metadata.
        """
        root_page_id = self.write_node_to_disk(root)
        self.write_tree_metadata(root_page_id)


class BTreeDisk(BTreeADT[T], CollectionADT[T], Generic[T]):
    """
    Disk Based B Tree: writes nodes to disk.
    Duplicate Keys are forbidden.
    Utilizes Pre-emptive fix Strategy for insert and deletion. (CLRS)
    Double Write (in place update)
    Nodes are written to disk via Page objects. The tree is stored in a Pagefile.
    There is a Page Manager interface, that allows you to load and save trees to disk.
    """
    def __init__(self, datatype: type, degree: int, pagefile: str) -> None:

        # initialize page file.
        self.page_manager = PageManager(pagefile, self._datatype, None, self._degree)
        self._root: None | BTreeNode =  None
        self._datatype = ValidDatatype(datatype)
        self._tree_keytype: None | type = None
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

    # ----- Loading A B-tree From Disk -----

    def load_from_disk(self):
        """Loads a tree from disk using page manager. will overwrite current in memory tree."""

        root_page_id, deg, total_nodes, dtype, ktype = self.page_manager.read_tree_metadata()

        if root_page_id is not None and total_nodes > 0:
            # delete existing tree
            self.clear()
            self._root = self.page_manager.read_node_from_disk(root_page_id)
            self._degree = deg
            self._datatype = ValidDatatype(dtype)
            self._tree_keytype = ValidDatatype(ktype)
        
        else:
            # no tree on disk.
            raise NodeEmptyError(f"Error: Tree is empty! check if pagefile exists")

    def save_tree_to_disk(self):
        """Saves a tree to disk in its current form."""
        if self._root is not None:
            self.page_manager.save_tree_to_disk(self._root)
        else:
            raise NodeEmptyError(f"Error: Tree Is Empty! No tree to save!")


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

    def load_child_from_disk(self, node):
        """Takes a node input and if it is not a node object already, loads it from disk."""
        if isinstance(node, BTreeNode):
            return node
        else:
            return self.page_manager.read_node_from_disk(node)

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
            child_page_id = node.children[idx]
            child = self.page_manager.read_node_from_disk(child_page_id)
            return self._recursive_search(child, key)

    def _node_search(self, key) -> Optional[tuple[BTreeNode, int]]:
        """
        Searches by key for the node that contains the key. 
        returns a tuple of the node and the key index. which can be accessed via the node.
        """

        # * load the root from disk:
        root_page_id, _, _, _ = self.page_manager.read_tree_metadata()
        root = self.page_manager.read_node_from_disk(root_page_id)

        return self._recursive_search(root, key)

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
            current = self.load_child_from_disk(current.children[current.num_keys - 1])
        return (current, current.num_keys - 1)

    def _successor(self, node: BTreeNode) -> tuple[BTreeNode, int]:
        """returns the succesor key  - the smallest key in the right subtree lareger than the specified key."""
        current = node
        while not current.is_leaf:
            current = self.load_child_from_disk(current.children[0])
        return (current, 0)

    def min(self) -> Optional[T]:
        """returns the minimum element in the b tree"""

        # read root from disk.
        root_page_id, _, _, _ = self.page_manager.read_tree_metadata()
        root = self.page_manager.read_node_from_disk(root_page_id)
        current = root

        # empty tree case:
        if current is None: return None

        # traverse
        while not current.is_leaf:
            current = self.load_child_from_disk(current.children[0])

        element: T = current.elements[0]
        return element

    def max(self) -> Optional[T]:
        """returns the max key (paired element) in the b tree"""

        # read root from disk.
        root_page_id, _, _, _ = self.page_manager.read_tree_metadata()
        root = self.page_manager.read_node_from_disk(root_page_id)
        current = root  

        # empty tree case:
        if current is None: return None

        # traverse
        while not current.is_leaf:
            current = self.load_child_from_disk(current.children[current.num_keys])

        last = current.num_keys - 1
        element: T = current.elements[last]
        return element

    # ----- Mutators -----
    def create_tree(self) -> None:
        """Creates a B tree and the root node"""
        # create root node in memory
        self._root = BTreeNode(self._datatype, self._degree, is_leaf=True)
        self._total_nodes +=1
        # write root to disk
        root_page_id = self.page_manager.write_node_to_disk(self._root)
        # record tree metadata (specific for root node)
        self.page_manager.write_tree_metadata(root_page_id)

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
            if node is not self._root:
                self.page_manager.write_node_to_disk(node)

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
        self._utils.set_keytype(key)
        key = Key(key)
        self._utils.check_btree_key_is_same_type(key)
        value = TypeSafeElement(value, self._datatype)

        # *empty tree case: insert into root node.
        if self._root is None:
            self._root = BTreeNode(self._datatype, self._degree, is_leaf=True)
            self._total_nodes += 1
            self._insert_non_full(self._root, key, value)
            # write to disk:
            root_page_id = self.page_manager.write_node_to_disk(self._root)
            self.page_manager.write_tree_metadata(root_page_id)
            return

        # * root is full
        if self._root.num_keys == self.max_keys:
            new_root = self.split_root()
            self._insert_non_full(new_root, key, value)
            # write to disk:
            root_page_id = self.page_manager.write_node_to_disk(self._root)
            self.page_manager.write_tree_metadata(root_page_id)
        # * insert into the root if not full.
        else:
            self._insert_non_full(self._root, key, value)
            # write to disk:
            root_page_id = self.page_manager.write_node_to_disk(self._root)
            self.page_manager.write_tree_metadata(root_page_id)

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

        # * write nodes to disk.
        self.page_manager.write_node_to_disk(child_node)
        self.page_manager.write_node_to_disk(new_sibling)
        self.page_manager.write_node_to_disk(parent_node)

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

    b = BTreeDisk(str, 5)
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
