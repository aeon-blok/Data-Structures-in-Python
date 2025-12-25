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
from faker import Faker
import logging
import logging.handlers
import traceback
import json

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
from utils.representations import TrieRepr
from utils.helpers import RandomClass
from utils.exceptions import *
from utils.constants import PAGE_SIZE, ALPHABET

from adts.collection_adt import CollectionADT
from adts.trie_adt import TrieADT

from ds.sequences.Stacks.array_stack import ArrayStack
from ds.primitives.arrays.dynamic_array import VectorArray, VectorView
from ds.maps.hash_table_with_chaining import ChainHashTable
from ds.maps.Sets.hash_set import HashSet
from ds.trees.tree_nodes import TrieNode
from ds.trees.tree_utils import TreeUtils

from user_defined_types.generic_types import (
    Index,
    ValidDatatype,
    ValidIndex,
    TypeSafeElement,
    PositiveNumber,
)

from user_defined_types.key_types import iKey, Key
from user_defined_types.tree_types import NodeColor, Traversal, PageID

# endregion


class Trie(TrieADT, CollectionADT[T]):
    """Trie Data Structure Implementation using Hash Map for children node."""
    def __init__(self) -> None:
        self._root: TrieNode = TrieNode(None)
        self._alphabet = VectorArray(26, str)
        self._alphabet.append_many(ALPHABET)
        self.word_count = 0

        # composed objects
        self._utils = TreeUtils(self)
        self._validators = DsValidation()
        self._desc = TrieRepr(self)


    @property
    def alphabet(self) -> Iterable:
        return self._alphabet
    
    @alphabet.setter
    def alphabet(self, input: Iterable) -> None:
        """
        changes the validation set for the trie. 
        (can change to a different language alphabet etc or numeric values)
        """
        self._alphabet = input

    def __str__(self) -> str:
        return self._desc.str_trie()
    
    def __repr__(self) -> str:
        return self._desc.repr_trie()
    
    # ----- Meta Collection ADT Operations -----

    def __len__(self) -> Index:
        """Counts the number of complete words in the trie."""
        return self.word_count

    def __contains__(self, word) -> bool:
        return self.search(word)

    def is_empty(self) -> bool:
        return self.word_count == 0

    def clear(self) -> None:
        self._root = TrieNode(None)
        self.word_count = 0

    def __iter__(self):
        """iterates over every word in the trie. returns results in lexographic order."""
        stack = ArrayStack(tuple)
        stack.push((self._root, ""))

        while stack:
            node, prefix = stack.pop()
            if node.is_end:
                yield prefix

            for char in reversed(self._alphabet):
                if char in node.children:
                    stack.push((node.children[char], prefix + char))

    # ----- Canonical ADT Operations -----
    # ----- Accessors -----
    def search(self, word: str) -> bool:
        """
        Its closer to a membership test than a search.... returns true if the word is in the trie, otherwise returns false.
        Traverses the trie - goes through each node - if the character does not exist in the word - return False, for unsuccesful.
        otherwise traverse to the next node.
        After consuming all characters, am I at a node marked is_end? if yes - return True for successful.
        """
        self._utils.check_is_string(word)
        word = word.lower()
        # * traverse through trie check if each character is in the trie, in the order specified.
        current = self._root
        for char in word:
            self._utils.assert_char_in_validation_set(char)
            if char not in current.children:
                return False
            current = current.children[char]
        return current.is_end

    def starts_with_prefix(self, prefix: str) -> bool:
        """
        Returns True if the prefix exists in the trie.
        similar to search but instead of the entire word - just uses a prefix.
        """
        self._utils.check_is_string(prefix)
        prefix = prefix.lower()
        current = self._root
        for char in prefix:
            self._utils.assert_char_in_validation_set(char)
            if char not in current.children:
                return False
            current = current.children[char]
        return True

    def _get_node(self, prefix: str):
        """given a prefix will return the end node for specified prefix."""
        self._utils.check_is_string(prefix)
        prefix = prefix.lower()
        current: TrieNode = self._root

        for char in prefix:
            if char not in current.children:
                return None
            current: TrieNode = current.children[char]
        return current

    def enumerate(self, prefix: str = ""):
        """
        Finds all words with a specific prefix & returns an iterable of the words.
        Utilizes DFS & recursion.
        """

        start = self._get_node(prefix)
        # existence check
        if start is None:
            return VectorArray(0, str)

        results = VectorArray(27, str)
        # holds all the characters after the prefix.
        path = VectorArray(27, str)

        def _recursive_dfs(node: TrieNode):
            """recursive dfs - appends complete words to the results list"""
            if node.is_end:
                results.append(prefix + "".join(path))

            for char, child in node.children.items():
                path.append(char)
                _recursive_dfs(child)
                path.delete(path.size-1) # removes the last appended char

        _recursive_dfs(start)
        return results

    def height(self) -> Index:
        """returns the tree height"""
        def _recursive_height(node: TrieNode):
            """recursively checks the children of a node, and compares the height to the total height of the tree so far."""
            if node.num_children == 0:
                return 1
            max_child_height = 0
            for _, child in node.children.items():
                max_child_height = max(max_child_height, _recursive_height(child))
            return 1 + max_child_height

        return 0 if self.is_empty() else _recursive_height(self._root)

    def get_children(self, node: TrieNode):
        """returns an iterable of all the children nodes of a specific node."""

        if not isinstance(node, TrieNode):
            raise NodeTypeError(f"Error: Invalid Node Type input. expected: {TrieNode}, got: {type(node)}")

        return node.children.values()

    # ----- Mutators -----
    def insert(self, word: str) -> bool:
        """
        Inserts a New Key into the Trie (usually a string(word)) -- O(M) (M=length of the string)
        Validate the characters of the word exist in the validation set(alphabet)
        Traverse the Trie Data Structure: If the character of the word doesnt exist in the trie - add it to the trie data structure.
        If the whole trie is traversed and all characters in the word already exist in the trie, return false.
        Returns True or False depending on whether the word already existed
        """

        # type check
        self._utils.check_is_string(word)

        word = word.lower()

        # * traverse trie
        current: TrieNode = self._root

        for char in word:
            # * validate every character in the key via the alphabet
            self._utils.assert_char_in_validation_set(char)
            # * create a new node if it doesnt exist in the trie.
            if char not in current.children:
                current.children.put(char, TrieNode(char))
            current = current.children[char]

        # if after traversing through tree - all the characters exist, the key already exists in the trie, return false.
        if current.is_end: 
            return False

        # * signify end of key / word
        current.is_end = True
        self.word_count += 1

        return True

    def delete(self, word) -> bool:
        """
        delete a stored word from the trie.
        Deletes nodes from the bottom until it hits a node that is required by another word.
        boolean return represents a successful or unsuccessful delete operation.
        """

        # type check
        self._utils.check_is_string(word)

        word = word.lower()

        # * traverse trie - and add characters to stack for deletion
        current = self._root
        delete_stack = ArrayStack(tuple)

        for char in word:
            # * validate every character in the key via the alphabet
            self._utils.assert_char_in_validation_set(char)
            if char not in current.children:
                return False
            delete_stack.push((current, char))  # (parent_node, child_node)
            current = current.children[char]

        # * Check if the whole word exists... (via is_end)
        if not current.is_end:
            return False

        # * from the bottom of trie - delete the characters found in both the word and the trie.
        # remove is_end boolean
        current.is_end = False
        self.word_count -= 1

        while delete_stack:
            parent, char = delete_stack.pop()
            child = parent.children[char]   # loads the child node.
            # Exit Condition: If this node is still needed by any word, STOP deleting. If either is true → deleting it would corrupt the trie.
            if child.is_end or child.num_children > 0:
                break
            # delete the node: a node is removable if it has no children & is_end == false
            del parent.children[char]
        return True


# ------------------------------- Main: Client Facing Code: -------------------------------


def main():
    fake = Faker()
    fake.seed_instance(202)
    word_set = {fake.word() for word in range(500)}
    words = list(word_set)
    amount = len(words) - 300
    words = words[amount:-1]
    print(f"Word Bank: Count: {len(words)}", sorted(words))
    test_word = fake.word()

    trie = Trie()
    print(trie)
    print(repr(trie))
    print(f"Is Trie empty?={trie.is_empty()}")
    print(f"Does Trie Contain the following word: {test_word}? result: {test_word in trie}")
    print(f"Number of Words in the Trie Currently={len(trie)}")

    print(f"Testing Insertion into trie")
    for i in words:
        trie.insert(i)
    print(trie)

    print(f"Testing Search of Trie")
    random_search_word = random.choice(words)
    print(f"Search For {random_search_word}. result={trie.search(random_search_word)}")
    print(f"Does Trie Contain the following word: {random_search_word}? result: {random_search_word in trie}")
    print(f"search for prefix: ")

    random_prefix = random_search_word[:len(random_search_word) // 2]
    print(f"Find all words with the following prefix: {random_prefix}")
    print(f"result={trie.enumerate(random_prefix)}")

    print(f"Testing Deletion of words in trie.")
    for i in words:
        trie.delete(i)
    print(trie)
    print(repr(trie))


if __name__ == "__main__":
    main()
