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
        return Page.SIZE - self._used_bytes

    def get_bytes(self) -> bytes:
        """Return a copy of the in-memory page bytes."""        
        return bytes(self.data)

    def modify_bytes(self, data: bytes) -> None:
        """Replace (inplace) the in-memory page bytes with new data."""

        #  overflow check
        if len(data) != Page.SIZE:
            raise DsInputValueError(f"Error: Bytes input exceeds the Page Capacity: {Page.SIZE}")

        self.data[:] = data

    def __str__(self) -> str:
        return self._desc.str_page()

    def __repr__(self) -> str:
        return self._desc.repr_page()


class PageManager:
    """
    Interface for writing nodes to disk, and reading nodes from disk.
    PageManager orchestrates serialization, disk writes, and tree structure.
    Utilizes a Free List that marks deleted pages and reuses them for new nodes.
    The Page Manager handles creating Nodes so it can assign page id's to them.
    Allocates Page ID's And Free's up deleted pages to be reused.
    """

    def __init__(self, location: str, datatype:Optional[type], keytype: Optional[type], degree: Optional[int]) -> None:
        self._auto_id: PageID = 1    
        self.page_table = ChainHashTable(BTreeNode)  # key = Page ID, value = Node
        self.pagefile = Path(location)
        self._datatype = datatype
        self._keytype = keytype
        self._degree = degree
        self._root_page_id = None
        self.free_list_head: Optional[PageID] = None
        self.free_list_cache: list[PageID] = []

        # control flow - empty pagefile, or existing pagefile.
        if self.pagefile.exists():
            if self.pagefile.stat().st_size != 0:
                self._load_existing_pagefile()
                self.load_tree_from_disk()
            else:
                if self._datatype is None or self._degree is None:
                    raise DsInputValueError(f"Error: Page Manager requires Datatype and Degree input parameters to be an actual value not none.")
                self._initialize_empty_pagefile(datatype, keytype, degree)
        else:
            if self._datatype is None or self._degree is None:
                raise DsInputValueError(f"Error: Page Manager requires Datatype and Degree input parameters to be an actual value not none.")
            self._initialize_empty_pagefile(datatype, keytype, degree)

    @property
    def keytype(self):
        return self._keytype

    @keytype.setter
    def keytype(self, value) -> None:
        self._keytype = value

    @property
    def root_page_id(self) -> Optional[PageID]:
        return self._root_page_id

    @root_page_id.setter
    def root_page_id(self, value: PageID) -> None:
        self._root_page_id = value

    # Initialize Page Manager
    def _initialize_empty_pagefile(self, datatype, keytype, degree):
        """If a pagefile doesnt exist. it will create a pagefile and add the metadata section (page 0)"""
        self.pagefile.touch()
        self._datatype = datatype
        self._keytype = keytype
        self._degree = degree
        self._root_page_id = None
        self.free_list_head: Optional[PageID] = None
        self.free_list_cache: list[PageID] = []
        self.initialize_metadata()

    def _load_existing_pagefile(self):
        """
        pagefile exists? load it and its required metadata
        We also need to derive the next auto_id from the pagefile itself. (to avoid pagefile collisions on load)
        """
        root_pid, freelist_head, deg, total_nodes, total_keys, dtype, ktype = self.read_tree_metadata()
        self.free_list_head: Optional[PageID] = freelist_head  # on disk implicit linked list
        self.free_list_cache: list[PageID] = []   # in memory (read tree metadata will mutate this.)
        self._datatype = ValidDatatype(dtype)
        self._keytype = ValidDatatype(ktype)
        self._degree = deg
        self._root_page_id = root_pid
        self.load_free_list_cache() # loads the cache on init.
        # we can derive the next auto id = pagefile_size // PAGE_SIZE
        # .stat() gives you filesystem info & .st_size is the total number of bytes in the file
        pagefile_size = self.pagefile.stat().st_size
        self._auto_id: PageID = pagefile_size // PAGE_SIZE

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

    # Page Id Management
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
            return page_id
        # no cache? check if on disk free list exists?
        elif self.free_list_head is not None:
            page_id = self.free_list_head
            page_bytes = self._read_page_bypass(page_id)
            next_free = int.from_bytes(page_bytes[:4], 'big')
            self.free_list_head = next_free if next_free != 0 else None
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

    # serializing nodes
    def _encode_node(self, node: BTreeNode):
        """
        Converts a Node into a fixed size byte representation. 
        and adds a page id and children page ids to the bytes.
        """

        # * validate node input.
        assert node.page_id is not None, f"Error: While trying to encode this node {node}, we discoverered it doesnt have a Page id!"

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
        # Node Page ID
        struct.pack_into("I", buffer, cursor, node.page_id)
        cursor += 4
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

            if cursor > PAGE_SIZE:
                raise DsOverflowError(f"Error: Node Serialization Exceeds Page Size")

        # elements
        for e in range(num_keys):
            element = node.elements[e]
            elem_bytes = pickle.dumps(element)
            elem_len = len(elem_bytes)
            struct.pack_into("H", buffer, cursor, elem_len)
            cursor += 2
            buffer[cursor:cursor+elem_len] = elem_bytes
            cursor += elem_len

            if cursor > PAGE_SIZE:
                raise DsOverflowError(f"Error: Node Serialization Exceeds Page Size")

        # children nodes (leaves dont have children so nothing to store...)
        if not node.is_leaf:
            for child_page_id in node.children:
                # packs the child page id into the buffer as an unsigned int.
                struct.pack_into("I", buffer, cursor, child_page_id)
                cursor += 4

                if cursor > PAGE_SIZE:
                    raise DsOverflowError(f"Error: Node Serialization Exceeds Page Size")

        return bytes(buffer)

    def _decode_node(self, page_bytes: bytes) -> BTreeNode:
        """
        Decodes bytes into a B Tree Node. 
        Must mirror Encode Node exactly
        """

        cursor = 0

        # header
        node_page_id = struct.unpack_from("I", page_bytes, cursor)[0]
        cursor += 4

        is_leaf = struct.unpack_from("B", page_bytes, cursor)[0]
        cursor += 1

        num_keys = struct.unpack_from("I", page_bytes, cursor)[0]
        cursor += 4

        # * create node object and assign the old page id to this newly created node
        node = BTreeNode(self._datatype, self._degree, is_leaf=bool(is_leaf))
        node.keytype = self._keytype
        node.page_id = node_page_id

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

    # storing pages to disk
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

    def create_node(self, datatype, degree, is_leaf) -> BTreeNode:
        """creates a B-Tree Node and assigns it a unique page_id."""
        new_node = BTreeNode(datatype, degree, is_leaf)
        new_node.page_id = self._allocate_page_id_via_free_list()
        return new_node

    def write_node_to_disk(self, node: BTreeNode) -> PageID:
        """
        writes a B-tree Node to disk, returns the page_id
        """

        # * validate input
        if node.page_id is None:
            raise DsInputValueError(f"Error: Node does not have an allocated page id. {node}")

        # collect page id from node.
        page_id = node.page_id

        # * encode node to bytes (with page id and children page id's)
        page_bytes = self._encode_node(node)
        page = Page(page_id, page_bytes)  # wrap in Page Object

        # * write page to disk
        self._store_page(page)

        return page_id

    def read_node_from_disk(self, page_id: PageID) -> BTreeNode:
        """reads bytes from disk, decodes the bytes into a node object"""

        # load page bytes
        page = self._load_page(page_id)
        page_bytes = page.get_bytes()

        # decode the page bytes into a node.
        # (this will create a new node with the same page id as the original node had.)
        node: BTreeNode = self._decode_node(page_bytes)

        assert node.page_id == page_id, f"Error: Node Page ID and input Page ID dont match..."

        return node

    def initialize_metadata(self) -> None:
        """
        Used on the first time creation of a pagefile - to initialize the metadata page (page 0)
        The root does not exist at this point. but will later.
        """
        buffer = bytearray(PAGE_SIZE)
        cursor = 0

        # root page id: 0 = No Tree or root
        struct.pack_into("I", buffer, cursor, 0)
        cursor += 4

        # free list head
        struct.pack_into("I", buffer, cursor, 0)
        cursor += 4

        # degree
        struct.pack_into("I", buffer, cursor, self._degree)
        cursor += 4

        # Total Nodes
        struct.pack_into("I", buffer, cursor, 0)
        cursor += 4

        # Total Keys
        struct.pack_into("I", buffer, cursor, 0)
        cursor += 4

        # datatype
        datatype_bytes_length = pickle.dumps(self._datatype)
        struct.pack_into("H", buffer, cursor, len(datatype_bytes_length))
        cursor += 2
        buffer[cursor:cursor+len(datatype_bytes_length)] = datatype_bytes_length
        cursor += len(datatype_bytes_length)

        # keytype
        keytype_bytes_length = pickle.dumps(self._keytype)
        struct.pack_into("H", buffer, cursor, len(keytype_bytes_length))
        cursor += 2
        buffer[cursor: cursor+len(keytype_bytes_length)] = keytype_bytes_length
        cursor += len(keytype_bytes_length)

        # record inside pagefile.
        self._store_page(Page(0, bytes(buffer)))

    def write_tree_metadata(self, root_page_id: PageID, total_nodes: int, total_keys: int) -> None:
        """
        Writes some simple metadata about the tree, including the root page id. 
        Which is essential for loading a tree from disk.
        requires us to pipe through the counters for total nodes and total keys in order to save them for up-to-date info when reloading tree.
        """
        buffer = bytearray(PAGE_SIZE)
        cursor = 0

        # root page id -
        self._root_page_id = root_page_id
        struct.pack_into("I", buffer, cursor, root_page_id)
        cursor += 4

        # free list head
        free_list_head = self.free_list_head if self.free_list_head else 0
        struct.pack_into("I", buffer, cursor, free_list_head)
        cursor += 4

        # degree
        struct.pack_into("I", buffer, cursor, self._degree)
        cursor += 4

        # total nodes
        struct.pack_into("I", buffer, cursor, total_nodes)
        cursor += 4

        # total keys
        struct.pack_into("I", buffer, cursor, total_keys)
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
        the start node for the free list (on disk)
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

        total_nodes = struct.unpack_from("I", buffer, cursor)[0]
        cursor += 4

        total_keys = struct.unpack_from("I", buffer, cursor)[0]
        cursor += 4

        datatype_bytes_length = struct.unpack_from("H", buffer, cursor)[0]
        cursor += 2
        self._datatype = pickle.loads(buffer[cursor:cursor+datatype_bytes_length])
        cursor += datatype_bytes_length

        keytype_bytes_length = struct.unpack_from("H", buffer, cursor)[0]
        cursor += 2
        self._keytype = pickle.loads(buffer[cursor:cursor+keytype_bytes_length])
        cursor += keytype_bytes_length

        return (root_page_id, self.free_list_head, self._degree, total_nodes, total_keys, self._datatype, self._keytype)

    def load_tree_from_disk(self) -> BTreeNode:
        """
        Loads the entire B-Tree from disk
        reads the metadata,
        recursively reads the children of the root and loads them into memory.
        returns the root node.
        """

        # metadata
        root_page_id, freelist_head, degree, total_nodes, total_keys, datatype, keytype = self.read_tree_metadata()
        self._degree: int = degree
        self._datatype: type = datatype
        self._keytype: type = keytype
        self._root_page_id = root_page_id
        self.free_list_head = freelist_head

        root = self.read_node_from_disk(root_page_id)

        return root

    def save_tree_to_disk(self, root: BTreeNode, total_nodes:int, total_keys:int) -> None:
        """
        Saves te entire B-Tree to Disk
        recursively writes all nodes starting from the root
        updates the tree metadata.
        """
        root_page_id = self.write_node_to_disk(root)
        self.write_tree_metadata(root_page_id, total_nodes, total_keys)


class BTreeDisk(BTreeADT[T], CollectionADT[T], Generic[T]):
    """
    Disk Based B Tree: writes nodes to disk.
    Duplicate Keys are forbidden.
    Utilizes Pre-emptive fix Strategy for insert and deletion. (CLRS)

    Nodes have a unique Page ID,
    children are stored as Page ID references.

    Nodes are written to disk (serialized) via Page objects. 

    The tree is stored in a Pagefile.
    There is a converted textfile that allows you to inspect the pagefile and its contents

   
    All node operations (read/write) happen through a page manager interface.

    Only nodes being traversed are loaded into memory (lazy loading).
    """
    def __init__(self, pagefile: str, datatype: Optional[type] = None, degree: Optional[int] = None) -> None:
        # this controls a large part of the b-tree
        self.page_manager = PageManager(pagefile, datatype, None, degree)
        # * existing tree found - load from disk.
        if self.page_manager.root_page_id is not None:
            self._root = self.page_manager.load_tree_from_disk()
            root_page_id, freelist_head, deg, total_nodes, total_keys, dtype, keytype = self.page_manager.read_tree_metadata()
            self._datatype = ValidDatatype(dtype)
            self._degree = PositiveNumber(deg)
            self.tree_keytype: None | type = keytype
        # * initialize new tree parameters
        else:
            if datatype is None or degree is None:
                raise DsInputValueError(f"Error: Input Parameters: Datatype & Degree are NoneType, but require input.")
            self._datatype = ValidDatatype(datatype)
            self._degree = PositiveNumber(degree)
            self.tree_keytype: None | type = None
            self._root: None | BTreeNode =  None

        self._total_nodes: int = 0
        self._total_keys: int = 0
        self.create_tree()

        # composed objects
        self._utils = TreeUtils(self)
        self._validators = DsValidation()
        self._desc = BTreeRepr(self)

    @property
    def datatype(self) -> type:
        return self._datatype

    @property
    def total_keys(self) -> int:
        """returns the total number of keys in the b tree"""
        return self._total_keys

    @property
    def tree_height(self) -> int:
        """the max tree height of the btree"""
        return self._utils.disk_btree_height_iterative(BTreeNode)

    @property
    def validate_tree(self) -> bool:
        return self._utils.disk_validate_btree()

    @property
    def bfs_view(self):
        return self._utils.disk_b_tree_bfs_view(BTreeNode)

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

    @root.setter
    def root(self, node: BTreeNode):
        """Sets the root node but also updates the tree metadata page id."""
        self._root = node
        self.page_manager.root_page_id = node.page_id
        assert self._root.page_id == self.page_manager.root_page_id, f"Error: root page id out of sync.... root pid={self._root.page_id} & Page manager root pid={self.page_manager.root_page_id}"


    @property
    def total_nodes(self) -> int:
        return self._total_nodes

    # ----- Loading A B-tree From Disk -----
    def inspect_pagefile(self):
        """reads the page file and inteprets the binary into string text."""

        if not self.page_manager.pagefile:
            return

        directory = self.page_manager.pagefile.parent
        filename = f"{self.page_manager.pagefile.stem}.txt"
        pagefile_log = directory / filename

        title = f"Disk B Tree Pagefile (converted to textfile for inspection)\n"
        fl_desc = f"the free list is a linked list of deleted page id's from the disk. when a new node is created it will utilize these disk blocks and ids for the next disk write.\n"

        with open(pagefile_log, "w", encoding="utf-8") as file:
            file.write(title)
            file.write(fl_desc)
            file_size = self.page_manager.pagefile.stat().st_size
            max_pages = file_size // PAGE_SIZE

            # decode metadata page (Page 0)
            try:
                root_page_id, free_list_head, deg, total_nodes, total_keys, dtype, ktype = self.page_manager.read_tree_metadata()
                file.write(f"\nPage 0 (Metadata):\n")
                file.write(f"Root Page ID: {root_page_id}\n")
                file.write(f"Free List Head: {free_list_head}\n")
                file.write(f"Degree: {deg}\n")
                file.write(f"Total Number of Nodes in Tree: {total_nodes}\n")
                file.write(f"Total Number of keys in the tree: {total_keys}\n")
                file.write(f"Tree DataType: (for elements): {dtype.__name__}\n")
                file.write(f"Key Type: (for keys): {ktype}\n")
                file.write(f"Free List (linked list): ")
                current = self.page_manager.free_list_head
                free_pages = []
                while current is not None:
                    free_pages.append(current)
                    page_data = self.page_manager._read_page_bypass(current)
                    next_free = int.from_bytes(page_data[:4], 'big')
                    current = next_free if next_free != 0 else None
                file.write(" -> ".join(map(str, free_pages)) + "\n\n")

            except Exception as e:
                file.write(f"Page 0: Metadata Decoding Failed! Error: {e}\n\n")

            for page_id in range(1, max_pages):
                try:
                    # only pages that are not in the free list will be inspected.
                    page = self.page_manager._load_page(page_id)
                except NodeDeletedError as e:
                    file.write(f"Page: {page_id} Has Been Deleted and can be located in the free list.\n")
                    continue
                except Exception as e:
                    file.write(f"Page: {page_id} load failed. Error: {e}\n")
                    continue
                try:
                    node = self.page_manager._decode_node(page.get_bytes())
                    file.write(f"Page: {page_id}\n")
                    file.write(f"Keys: {node.keys}\n")
                    file.write(f"Elements: {node.elements}\n")
                    file.write(f"Children: {node.children if not node.is_leaf else 'Leaf Node'}\n")
                except Exception as e:
                    file.write(f"Page: {page_id}: couldnt decode.... Error: {e}\n")

        print(f"Inspection of Pagefile written to: {pagefile_log}")

    def convert_page_id_to_node(self, input: BTreeNode | PageID) -> Optional[BTreeNode]:
        """Converts a Page ID into a Node, if the item is already a node, it just returns it immediately."""
        if isinstance(input, BTreeNode):
            return input
        if isinstance(input, PageID):
            if input in self.page_manager.free_list_cache:
                raise NodeExistenceError(f"Error: Page ID: {input} is in free list and cannot be utilized.")
            else:
                return self.page_manager.read_node_from_disk(input)
        else:
            raise DsTypeError(f"Error: Expected Node or Page ID got: {type(input)}")

    def extract_page_id(self, input: BTreeNode | PageID) -> PageID:
        """Checks whether the input is a node, if it is extracts its page id"""
        if isinstance(input, PageID):
            return input
        if isinstance(input, BTreeNode):
            return input.page_id

    def write_node_to_disk(self, node) -> Optional[PageID]:
        """takes a node input, validates is and then writes it to disk."""

        if not isinstance(node, BTreeNode):
            raise DsTypeError(f"Error: Before writing to disk. the input must be an Actual Node object.")

        if node == self._root:
            root_page_id = self.write_root_to_disk()
            return root_page_id
        else:
            page_id = self.page_manager.write_node_to_disk(node)

            return page_id

    def delete_node_from_disk(self, page_id: PageID) -> None:
        """marks a page as a free page, and allows it to be used and overwritten by new inserted pages."""
        # * validate input
        self.page_manager.free_page_id(page_id)
        if page_id != self.page_manager.root_page_id:
            self._total_nodes -= 1
        self.page_manager.write_tree_metadata(self.page_manager.root_page_id, self._total_nodes, self._total_keys)

    def load_root_from_disk(self):
        """loads the root node from disk"""
        root_page_id, freelist_head, deg, total_nodes, total_keys, dtype, ktype = self.page_manager.read_tree_metadata()
        root = self.page_manager.read_node_from_disk(root_page_id)
        return root

    def write_root_to_disk(self) -> Optional[PageID]:
        """Writes spefically the root node to disk and updates the metadata"""

        # write root to disk (this returns the page id)
        root_page_id = self.page_manager.write_node_to_disk(self._root)
        self.page_manager.root_page_id = root_page_id

        # record tree metadata (specific for root node) --we need the root node to always represent accurate metadata information in the pagefile.
        # (this is used to load a b-tree)
        self.page_manager.write_tree_metadata(self.page_manager.root_page_id, self._total_nodes, self._total_keys)

        return root_page_id

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
    def _recursive_search(self, node: BTreeNode, key) -> Optional[tuple[BTreeNode, int]]:
        """recursively searches the whole tree until a match is found or None is returned."""
        # * empty tree case: existence check

        # init vars
        # will check if its a node or page id - if its a page id will load it from disk.
        current_node = self.convert_page_id_to_node(node)
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
            child_page_id = current_node.children[idx]
            child = self.convert_page_id_to_node(child_page_id)
            return self._recursive_search(child, key)

    def _node_search(self, key) -> Optional[tuple[BTreeNode, int]]:
        """
        Searches by key for the node that contains the key. 
        returns a tuple of the node and the key index. which can be accessed via the node.
        """

        # * load the root from disk:
        root = self.load_root_from_disk()

        return self._recursive_search(root, key)

    def search(self, key) -> T | None:
        """
        public facing method
        Searches for the specified key in the B tree and returns the element value.
        """

        if self.tree_keytype is None:
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
        current = self.convert_page_id_to_node(node)
        last = current.num_keys - 1
        while not current.is_leaf:
            # traverse to the rightmost child.
            current = self.convert_page_id_to_node(current.children[last])
        return (current, last)

    def _successor(self, node: BTreeNode) -> tuple[BTreeNode, int]:
        """returns the succesor key  - the smallest key in the right subtree lareger than the specified key."""
        current = self.convert_page_id_to_node(node)
        while not current.is_leaf:
            current = self.convert_page_id_to_node(current.children[0])
        return (current, 0)

    def min(self) -> Optional[T]:
        """returns the minimum element in the b tree"""

        # read root from disk.
        current = self.load_root_from_disk()

        # empty tree case:
        if current is None: return None

        # traverse
        while not current.is_leaf:
            current = self.convert_page_id_to_node(current.children[0])

        element: T = current.elements[0]
        return element

    def max(self) -> Optional[T]:
        """returns the max key (paired element) in the b tree"""

        # read root from disk.
        current = self.load_root_from_disk()  

        # empty tree case:
        if current is None: return None

        # traverse
        while not current.is_leaf:
            current = self.convert_page_id_to_node(current.children[current.num_keys])

        last = current.num_keys - 1
        element: T = current.elements[last]
        return element

    # ----- Mutators -----
    def create_tree(self) -> None:
        """Creates a B tree and the root node"""
        # create root node in memory
        self._root = self.page_manager.create_node(self._datatype, self._degree, is_leaf=True)
        self._total_nodes +=1
        root_pid = self.write_root_to_disk()
        self.load_root_from_disk()

    def _insert_non_full(self, node, key, value):
        """
        helper method: inserts into a non full node.
        """

        node = self.convert_page_id_to_node(node)
        idx = node.num_keys - 1  # the last key index

        # * leaf case: - insert key into node. (no further action needed)
        if node.is_leaf:
            # Linear Scan: find the correct index for the key.
            while idx >= 0 and key < node.keys[idx]:
                idx -= 1
            # insert key and value into the node
            node.keys.insert(idx+1, key)
            node.elements.insert(idx+1, value)
            self._total_keys += 1
            self.page_manager.write_tree_metadata(self.page_manager.root_page_id, self._total_nodes, self._total_keys)
            self.page_manager.write_node_to_disk(node)
            node = self.convert_page_id_to_node(node.page_id)
            self._utils.assert_root_pid_in_sync()

        # * internal node - find the child where key belongs
        else:
            # traverse backwards through keys until new key is greater than current key
            while idx >= 0 and key < node.keys[idx]:
                idx -= 1
            # move forward 1 step to get the correct index for the new key.
            idx += 1
            # * split child if its full
            # with the disk variant - first we must convert all child page id's to a real node.
            child_page_id = node.children[idx]
            child = self.convert_page_id_to_node(child_page_id)
            if child.num_keys == self.max_keys:
                self.split_child(node, idx)
                # if new key is larger -- it goes in the right child. otherwise goes in the left child.
                if key > node.keys[idx]:
                    idx += 1
            # insert key and value into the correct child slot.
            child_page_id = node.children[idx]
            child = self.convert_page_id_to_node(child_page_id)
            self._insert_non_full(child, key, value)

    def insert(self, key, value: T) -> None:
        """
        Public Facing Insert Method: Inserts a Key Value Pair into an existing leaf node.
        Overflow Rule: If the node is full - performs a split child/root operation. (on every full node you encounter traversing the tree.)
        Fix Then Insert Strategy: Utilizes the strategy employed by CLRS -
        Nodes are pre-emptively checked for number of keys and split if full.
        this allows the insertion to be completed in a single traversal down the tree.
        rather than having to go back up the tree to fix nodes that violate the b tree properties.
        Root writes: because root may change (split) and metadata must stay consistent.
        the root is key to loading and saving B trees.
        """

        # * validate inputs
        key = Key(key)
        self._utils.disk_set_keytype(key)
        self._utils.check_btree_key_is_same_type(key)
        value = TypeSafeElement(value, self._datatype)
        self._root = self.load_root_from_disk()

        # *empty tree case: create root node, and then insert into root node.
        if self._root.num_keys == 0:
            self._insert_non_full(self._root, key, value)   # write happens inside
            return

        # * root is full
        if self._root.num_keys == self.max_keys:
            self.write_root_to_disk()
            self._root = self.split_root()
            self._insert_non_full(self._root, key, value)
            # write to disk:
            self.write_root_to_disk()

        # * insert into the root if not full.
        else:
            self._insert_non_full(self._root, key, value)
            # write to disk:
            self.write_root_to_disk()

    def delete(self, key) -> None:
        """
        public delete method - utilizes recursive deletion.
        Fix then Delete Strategy: Utilizes pre-emptive checking to ensure that every child has over the min number of keys. 
        which allows us to delete a key without extra operations.
        If they do not have the required number of keys (t) then perform a borrow or merge operation
        """

        # * validate input
        key = Key(key)
        self._utils.check_btree_key_is_same_type(key)
        self._root = self.load_root_from_disk()

        print(f"\nB-tree delete: {key}")
        # * Empty tree Case:
        if self._root.num_keys == 0:
            print(f"btree is empty - no further action")
            return

        self._recursive_delete(self._root, key)
        self._root = self.load_root_from_disk()

        # * root edge case: root is empty & has exactly 1 child. promote child to root and delete old root.
        if self._root.num_keys == 0:
            if not self._root.is_leaf:
                print(f"ROOT EDGE CASE: root is empty & has exactly 1 child. promote child to root and delete old root.")
                # store root page id to free up later.
                print(f"root pid: {self._root.page_id}, page manager root pid = {self.page_manager.root_page_id}")
                # assert self._root.page_id == self.page_manager.root_page_id, f"Error: root pid and page manager root pid dont match!"
                old_root = self._root
                old_root_pid = self.write_node_to_disk(old_root)
                # promote only child to be new root.
                self._root = self.convert_page_id_to_node(self._root.children[0])
                self.write_root_to_disk()
                # free up the old root page id.
                self.delete_node_from_disk(old_root_pid)
            else:
                # tree is empty: (root is a leaf with 0 keys)
                self.write_root_to_disk()

    def _case_1_leaf_node_contains_key(self, parent_node, idx) -> None:
        """
        Case 1A: current has min + 1 keys:
        You don’t need to reload the parent/leaf node in Case 1. no chance of stale references
        """
        print(f"CASE 1: Entering Case 1")
        self._root = self.load_root_from_disk()

        if parent_node.num_keys > self.min_keys:
            print(f"Deleting Key: {parent_node.keys[idx]}")
            parent_node.keys.delete(idx)
            parent_node.elements.delete(idx)
            self._total_keys -= 1
            parent_pid = self.write_node_to_disk(parent_node)
            self._utils.assert_root_pid_in_sync()
            self.page_manager.write_tree_metadata(self._root.page_id, self._total_nodes, self._total_keys)
        elif parent_node == self._root:
            print(f"ROOT CASE: Node is the Root and the only node left: deleting Key: {parent_node.keys[idx]}")
            parent_node.keys.delete(idx)
            parent_node.elements.delete(idx)
            self._total_keys -= 1
            parent_pid = self.write_node_to_disk(parent_node)    # will auto check if its the root
            self.page_manager.write_tree_metadata(parent_pid, self._total_nodes, self._total_keys)
        else:
            raise KeyInvalidError(f"Error: Case 1: Key not found.")

    def _case_2_internal_node_contains_key(self, parent_node, idx, key) -> None:
        """
        Case 2A: child i has the min + 1 required keys
        Case 2B: child i has min keys, and its right sibling has min + 1 keys
        Case 2C: both child and sibling have min keys. (cant borrow need to merge.)
        """
        child = self.convert_page_id_to_node(parent_node.children[idx])
        right_sibling = self.convert_page_id_to_node(parent_node.children[idx+1]) if idx + 1 < parent_node.num_keys + 1 else None
        left_sibling = self.convert_page_id_to_node(parent_node.children[idx - 1]) if idx > 0 else None

        if child.num_keys >= self._degree:
            print(f"CASE 2A: Entering Case 2A: child pointer={child}")

            # * find predecessor:
            pred, pred_idx = self._predecessor(child)
            pred_key: iKey = pred.keys[pred_idx]
            pred_element: T = pred.elements[pred_idx]
            print(f"predecessor: {pred_key} and {pred}")

            # * replace parent key / element with predecessor key.
            parent_node.keys[idx] = pred_key
            parent_node.elements[idx] = pred_element

            # * after swapping parent and predecessor key / element - write to disk to persist changes.
            # ensure child is not a stale reference by reloading node from page id. same for parent
            parent_node_pid = self.write_node_to_disk(parent_node)
            parent_node = self.convert_page_id_to_node(parent_node_pid)
            child = self.convert_page_id_to_node(parent_node.children[idx])

            assert child.num_keys >= self._degree, f"Error: Case 2A: Child doesnt have t keys."
            print(f"Case 2A: recursively entering child with pred key")
            self._recursive_delete(child, pred_key)
            return

        elif child.num_keys == self.min_keys and right_sibling is not None and right_sibling.num_keys >= self._degree:
            print(f"CASE 2B: Entering Case 2B: child pointer={child}, right sibling={right_sibling}")
            # find successor:
            succ, succ_idx = self._successor(right_sibling)
            succ_key = succ.keys[succ_idx]
            succ_element = succ.elements[succ_idx]
            print(f"succesor: {succ_key}, {succ}")
            # replace parent key with succ key
            parent_node.keys[idx] = succ_key
            parent_node.elements[idx] = succ_element

            # write updated keys to disk and refresh references
            parent_node_pid = self.write_node_to_disk(parent_node)
            parent_node = self.convert_page_id_to_node(parent_node_pid)
            right_sibling = self.convert_page_id_to_node(parent_node.children[idx+1])

            assert right_sibling.num_keys >= self._degree, f"Error: Case 2B: Child doesnt have t keys."
            print(f"Case 2B: recursively entering right sibling with succ key")
            self._recursive_delete(right_sibling, succ_key)
            return

        # * Case 2C: both child i and siblings have min keys. (cant borrow need to merge.)
        elif child.num_keys == self.min_keys: 
            print(f"CASE 2C: Entering Case 2C child={child}, right={right_sibling}, left={left_sibling}")
            # merge right sibling into child
            if right_sibling is not None and right_sibling.num_keys == self.min_keys:
                print(f"merge right into child operation:")
                child_pid, parent_pid = self.merge_right_into_child(parent_node, idx)
                parent_node = self.convert_page_id_to_node(parent_pid)
                merged_child = self.convert_page_id_to_node(child_pid)
                print(f"merged={merged_child}")
                print(f"Merged Child Keys = {merged_child.keys}")
                assert merged_child.num_keys == self.max_keys, f"Error: Case 2C: Merged Child should have Max number of keys. (CLRS)"
                assert merged_child.num_keys >= self._degree, f"Error: Case 2C: Child doesnt have t keys."
                print(f"Entering recursive delete on merged child.")
                self._recursive_delete(merged_child, key)
                return
            # * Last Child Edge Case: merge child into left sibling (affects index order)
            elif left_sibling is not None and left_sibling.num_keys == self.min_keys:
                left_pid, parent_pid = self.merge_with_left(parent_node, idx)
                print(f"merge child with left operation:")
                parent_node = self.convert_page_id_to_node(parent_pid)
                merged_node = self.convert_page_id_to_node(left_pid)
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

    def _case_3_internal_node_does_not_contain_key(self, parent_node, idx, key) -> None:
        """
        Case 3A: Child i has min (t-1) keys, but sibling has min + 1 keys -- (borrow from sibling)
        Borrow median key from parent. and swap this with sibling.
        Case 3B:  Child and siblings have min keys (merge child with sibling)
        we need to move a key from current node to become the median key for this new merged node.
        Merging with right is preferable because it maintains index order.
        """

        # init family unit
        child = self.convert_page_id_to_node(parent_node.children[idx])
        left_sibling = self.convert_page_id_to_node(parent_node.children[idx - 1]) if idx > 0 else None
        right_sibling = self.convert_page_id_to_node(parent_node.children[idx+1]) if idx + 1 < parent_node.num_keys + 1 else None

        print(f"CASE 3: entering case 3: child={child}, left={left_sibling}, right={right_sibling}")

        if child.num_keys == self.min_keys:
            # * Case 3A: Child i has min keys, but sibling has min + 1 keys -- (borrow from sibling)
            # Case 3A: borrow key from left sibling
            if left_sibling is not None and left_sibling.num_keys > self.min_keys:
                print(f"Case 3A: borrow from left. performing borrow left op")
                self.borrow_left(parent_node, idx)
                child = self.convert_page_id_to_node(parent_node.children[idx])
                print(f"child={child} Entering recursive delete on child")
                self._recursive_delete(child, key)

            # Case 3A: borrow key from right sibling
            elif right_sibling is not None and right_sibling.num_keys > self.min_keys:
                print(f"Case 3A: borrow from right. performing borrow right op")
                self.borrow_right(parent_node, idx)
                child = self.convert_page_id_to_node(parent_node.children[idx])
                print(f"child={child} Entering recursive delete on child")
                self._recursive_delete(child, key)

            # * Case 3B:  Child and siblings have min keys (merge child with sibling)
            elif right_sibling is not None and right_sibling.num_keys == self.min_keys:
                print(f"Case 3B: Merge Right -- performing merge right into child op")
                child_pid, parent_pid = self.merge_right_into_child(parent_node, idx)
                parent_node = self.convert_page_id_to_node(parent_pid)
                merged_child = self.convert_page_id_to_node(parent_node.children[idx])
                assert merged_child.num_keys == self.max_keys, f"Error: Case 3B: Merged Child should have Max number of keys. (CLRS)"
                print(f"merged child={merged_child} Entering recursive delete on merged child")
                self._recursive_delete(merged_child, key)

            # merge with left sibling (if it exists.)
            elif left_sibling is not None and left_sibling.num_keys == self.min_keys:
                print(f"Case 3B: Merge Left -- performing merge child into left op")
                left_pid, parent_pid = self.merge_with_left(parent_node, idx)
                parent_node = self.convert_page_id_to_node(parent_pid)
                merged_node = self.convert_page_id_to_node(parent_node.children[idx-1])
                assert merged_node.num_keys == self.max_keys, f"Error: Case 3B: Merged Node (left sibling) should have Max number of keys. (CLRS)"
                print(f"Merged Child Keys={merged_node.keys}")
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
        parent_node = self.convert_page_id_to_node(node)

        if parent_node == self._root:
            print(f"Entering Recursive Delete on Root: ROOT={parent_node} is_leaf: {node.is_leaf}")
        else:
            print(f"Entering Recursive Delete: node={parent_node} is_leaf: {node.is_leaf}")

        # * Linear Scan: traverse through keys and find the key...
        while idx < parent_node.num_keys and key > parent_node.keys[idx]:
            idx += 1  # increment counter
        print(f"keys = {parent_node.keys}")
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

        for i in self._utils.disk_b_tree_inorder():
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

        print(f"Splitting Root: ")
        # store old root first
        old_root = self._root
        old_root_page_id = self.write_node_to_disk(old_root)
        print(f"old root leaf? {old_root.leaf}")

        # allocate new root (will allocate page id automatically)
        new_root = self.page_manager.create_node(self._datatype, self._degree, is_leaf=False)
        self._total_nodes += 1
        # make the old root a child of the new node.
        new_root.children.insert(0, old_root_page_id)
        print(f"new root children = {new_root.children}")
        # * new node becomes the new root.
        self._root = new_root
        print(f"self.root children = {self._root.children}")
        root_page_id = self.write_root_to_disk()
        # Split the first child of new_node, (which is the old root)
        self.split_child(self._root, 0)
        self._root = self.convert_page_id_to_node(root_page_id)
        return self._root

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

        print(f"Splitting Child: ")
        # child - retains the first half of the keys
        parent_node = self.convert_page_id_to_node(parent_node)
        child_node: BTreeNode = self.convert_page_id_to_node(parent_node.children[index])
        print(f"parent={parent_node}, child={child_node}")

        # * we create a new sibling - it will inherit its leaf status from its other sibling (the child)
        new_sibling: BTreeNode = self.page_manager.create_node(self._datatype, self._degree, is_leaf=child_node.leaf)
        self._total_nodes += 1

        # median key
        med_idx = self._degree - 1
        median_key = child_node.keys[med_idx]
        median_element = child_node.elements[med_idx]

        # * collect the largest keys and elements from the child. and give them to the sibling.
        # moves the minimum number of keys necessary to the new node
        for idx in range(self.min_keys):
            # copies over the keys that are higher than the min number of keys.
            new_sibling.keys.append(child_node.keys[idx + self._degree])
            new_sibling.elements.append(child_node.elements[idx + self._degree])
        # copy over children also (for disk variants we only allow Page ID references not actual nodes.)
        if not child_node.is_leaf:
            for idx in range(self._degree):
                child_page_id = child_node.children[idx + self._degree]
                new_sibling.children.append(child_page_id)

        # * delete the second half of keys and children from child node.
        for _ in range(self.min_keys):
            child_node.keys.delete(self._degree)
            child_node.elements.delete(self._degree)
        if not child_node.is_leaf:
            for _ in range(self._degree):
                child_node.children.delete(self._degree)

        # * relink parent and new child. (and add promoted key)
        # add new sibling page id (not the node) to parent's children list
        new_sibling_page_id = new_sibling.page_id
        parent_node.children.insert(index + 1, new_sibling_page_id)

        # now insert promoted median key. (t-1)
        parent_node.keys.insert(index, median_key)
        parent_node.elements.insert(index, median_element)

        # remove median key from child node.
        child_node.keys.delete(self._degree-1)
        child_node.elements.delete(self._degree-1)

        # * write nodes to disk.
        child_pid = self.page_manager.write_node_to_disk(child_node)
        new_sibling_pid = self.page_manager.write_node_to_disk(new_sibling)
        parent_pid = self.page_manager.write_node_to_disk(parent_node)
        self.page_manager.write_tree_metadata(self.page_manager.root_page_id, self._total_nodes, self._total_keys)

    def merge_right_into_child(self, parent_node, idx: Index) -> tuple[PageID, PageID]:
        """
        Merges the right sibling into the child node.
        the child is removed from the parent children list.
        """
        child = self.convert_page_id_to_node(parent_node.children[idx])
        right_sibling_page_id = parent_node.children[idx+1]
        right_sibling = self.convert_page_id_to_node(right_sibling_page_id)

        # move median key down from parent
        median_key = parent_node.keys[idx]
        median_element = parent_node.elements[idx]
        child.keys.append(median_key)
        child.elements.append(median_element)

        # merge right sibling INTO child
        for i in right_sibling.keys: 
            child.keys.append(i)
        for i in right_sibling.elements: 
            child.elements.append(i)
        if not right_sibling.is_leaf:
            for i in right_sibling.children:
                i = self.extract_page_id(i) 
                child.children.append(i)

        # remove median key / element from parent
        parent_node.keys.delete(idx)
        parent_node.elements.delete(idx)

        # remove right sibling Page ID from parent
        parent_node.children.delete(idx + 1)

        child_pid = self.write_node_to_disk(child)
        parent_pid = self.write_node_to_disk(parent_node)
        self.delete_node_from_disk(right_sibling_page_id)

        return (child_pid, parent_pid)

    def merge_with_left(self, parent_node, idx: Index) -> tuple[PageID, PageID]:
        """
        Merges a child node into its left sibling. for this it uses its parent's node's median key. (its passed down)
        the child is then removed from the parent...
        """
        child_page_id = parent_node.children[idx]
        child = self.convert_page_id_to_node(child_page_id)
        left_sibling = self.convert_page_id_to_node(parent_node.children[idx - 1])

        # move parent key down to become median key for new merged node.
        median_key = parent_node.keys[idx - 1]
        median_element = parent_node.elements[idx - 1]
        # append median key to the array.
        left_sibling.keys.append(median_key)
        left_sibling.elements.append(median_element)

        # now append the child keys INTO the Left sibling. and elements.
        # leaf check to avoid stale references on disk.
        for i in child.keys: left_sibling.keys.append(i)
        for i in child.elements: left_sibling.elements.append(i)
        if not child.is_leaf:
            for i in child.children: 
                i = self.extract_page_id(i)
                left_sibling.children.append(i)

        # * delete median key / element from parent.
        parent_node.keys.delete(idx-1)
        parent_node.elements.delete(idx-1)

        # remove child from parent.
        parent_node.children.delete(idx)

        left_pid = self.write_node_to_disk(left_sibling)
        parent_pid = self.write_node_to_disk(parent_node)
        self.delete_node_from_disk(child_page_id)
        return (left_pid, parent_pid)

    def borrow_left(self, parent_node, idx: Index) -> None:
        """
        Borrows the last key / element from the left sibling and moves it up to the parent.
        then moves the corresponding parent key / element down to the RIGHT child
        assumes the nodes involved are internal nodes.
        The key separating the two nodes is at index (idx - 1)
        Borrow is in essence a rotation, applied to two keys.
        """

        child = self.convert_page_id_to_node(parent_node.children[idx])
        left_sibling = self.convert_page_id_to_node(parent_node.children[idx-1])

        # move parent key down into child:
        child.keys.insert(0, parent_node.keys[idx-1])
        child.elements.insert(0, parent_node.elements[idx-1])

        # move last key from left sibling up into parent
        last = left_sibling.num_keys - 1    # is this correct?
        parent_node.keys[idx-1] = left_sibling.keys[last]
        parent_node.elements[idx-1] = left_sibling.elements[last]

        # move child pointer from left sibling to child children array.
        if not left_sibling.is_leaf:
            last_left_child_pid: PageID = self.extract_page_id(left_sibling.children[last])
            child.children.insert(0, last_left_child_pid)
            left_sibling.children.delete(last)

        # delete key from left sibling
        left_sibling.keys.delete(last)
        left_sibling.elements.delete(last)

        # write to disk
        left_pid = self.write_node_to_disk(left_sibling)
        child_pid = self.write_node_to_disk(child)
        parent_pid = self.write_node_to_disk(parent_node)

    def borrow_right(self, parent_node, idx: Index) -> None:
        """
        Borrows the first key / element from the right sibling and moves it up to the parent.
        Then Moves the Corresponding parent key / element down into the LEFT child.
        parent key --> child key.
        right_sibling key --> parent key
        Borrow is in essence a rotation, applied to two keys.
        """
        child = self.convert_page_id_to_node(parent_node.children[idx])
        right_sibling = self.convert_page_id_to_node(parent_node.children[idx+1])

        # move key from parent down into child
        child.keys.append(parent_node.keys[idx])  
        child.elements.append(parent_node.elements[idx])

        # move first key from right sibling up into parent
        parent_node.keys[idx] = right_sibling.keys[0]
        parent_node.elements[idx] = right_sibling.elements[0]

        # move child pointer from right sibling to child children array.
        # ONLY if internal node. (leaves dont have children)
        if not right_sibling.is_leaf:
            first_right_child_pid = self.extract_page_id(right_sibling.children[0])
            child.children.append(first_right_child_pid)
            right_sibling.children.delete(0)

        # delete first key from right sibling.
        right_sibling.keys.delete(0)
        right_sibling.elements.delete(0)

        right_pid = self.write_node_to_disk(right_sibling)
        child = self.write_node_to_disk(child)
        parent = self.write_node_to_disk(parent_node)


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


    print(f"\nTesting Disk Based B Tree")
    pagefile_location = r"J:\CODE\Python_Data_Structures_2025\src\ds\trees\B_Trees\Save_Dir\diskb.page"
    diskb = BTreeDisk(pagefile_location, str, 5)
    print(diskb)
    print(repr(diskb))

    print(f"\nTesting Insert functionality of Btree")
    for i, item in zip(keys, random_data):
        print(diskb)
        diskb.insert(i, item)
        print(repr(diskb))

    print(diskb)

    print(f"\nTesting Search functionality of Btree: key:0 = {diskb.search(0)}")
    print(f"Testing Contains functionality: key:23423425 in disk B tree? {23423425 in diskb}")
    print(f"Is Disk B-Tree Empty?: {diskb.is_empty()}")

    print(f"\nTesting Delete functionality of Btree")
    print(diskb)
    print(repr(diskb))
    shuffled_keys = list(range(15))
    random.shuffle(shuffled_keys)
    for key in shuffled_keys:
        diskb.delete(key)
        print(diskb)
        print(repr(diskb))

    # print(diskb)
    diskb.inspect_pagefile()


    # b = BTreeDisk(str, 5)
    # print(f"Does key 3 exist? {'Yes' if 3 in b else 'No'}")

    # print(f"\nTesting Insert functionality of Btree")
    # for i, item in zip(keys, random_data):
    #     b.insert(i, item)

    # print(repr(b))
    # print(b)
    # b.validate_tree

    # print(f"\nTesting Node repr")
    # print(b._root)
    # print(repr(b.root))

    # print(f"\nTesting Search Functionality: key:25 = {b.search(25)}")
    # print(f"Testing Search on a non existent key: key:200 = {b.search(200)}")

    # min_val = b.min()
    # max_val = b.max()
    # print(f"Min element: {min_val}")
    # print(f"Max element: {max_val}")

    # print("\nTesting __contains__ and __len__...")
    # print(f"Does key 3 exist? {'Yes' if 3 in b else 'No'}")
    # print(f"Total keys in tree: {len(b)}")

    # # ---------- Traverse ----------
    # print("\nTesting traversal...")
    # print(b.traverse("keys"))
    # print(b.traverse("elements"))
    # print(b.traverse("tuple"))

    # print(f"\nTesting Delete functionality...")
    # print(b)
    # b.validate_tree
    # print(f"Testing randomized deletion")
    # shuffled_keys = list(range(30))
    # random.shuffle(shuffled_keys)
    # for key in shuffled_keys:
    #     b.delete(key)
    #     print(b)
    # b.validate_tree

    # # ---------- Type Checking ----------
    # print("\nTesting type validation...")
    # try:
    #     b.insert(6, RandomClass("alyyyllgfdgfd"))  # invalid element type
    # except Exception as e:
    #     print(f"Caught expected type error: {e}")


if __name__ == "__main__":
    main()
