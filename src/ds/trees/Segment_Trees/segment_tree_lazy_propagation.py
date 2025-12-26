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
from utils.representations import SegmentTreeRepr
from utils.helpers import RandomClass
from utils.exceptions import *

from ds.primitives.arrays.dynamic_array import VectorArray, VectorView
from ds.trees.tree_utils import TreeUtils

from user_defined_types.generic_types import (
    Index,
    ValidDatatype,
    ValidIndex,
    TypeSafeElement,
    PositiveNumber,
)

from user_defined_types.tree_types import NodeColor, Traversal, PageID, SegmentOperator

# endregion
