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
from collections.abc import Sequence

# endregion


# region custom imports
from utils.custom_types import T, K, Key
from utils.validation_utils import DsValidation
from utils.exceptions import *
from utils.helpers import Ansi

if TYPE_CHECKING:
    from adts.collection_adt import CollectionADT
    from adts.map_adt import MapADT
    from adts.sequence_adt import SequenceADT

from ds.primitives.arrays.dynamic_array import VectorArray

# endregion


class MapUtils:
    """A collection of Utilities for Map Data Structures (hash tables, sets etc)"""
    def __init__(self, map_obj) -> None:
        self.obj = map_obj

        # composed objects
        self._hashfunc = None
        self._probefunc = None

    def select_hash_function(self):
        pass

    def select_probe_function(self):
        pass

    


