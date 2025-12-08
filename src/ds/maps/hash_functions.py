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
from dataclasses import dataclass, field
import random
import os, hashlib, math, itertools

# endregion

# region custom imports
from user_defined_types.generic_types import T, K
from user_defined_types.hashtable_types import HashCodeType, CompressFuncType, BitMask
from user_defined_types.key_types import iKey
from utils.validation_utils import DsValidation
from utils.exceptions import *
from utils.helpers import Ansi
from utils.constants import MIN_HASHTABLE_CAPACITY

if TYPE_CHECKING:
    from adts.collection_adt import CollectionADT
    from adts.map_adt import MapADT
    from adts.sequence_adt import SequenceADT


# endregion


# ------------------- Code Interface -----------------------

"""
How to use this stuff in your code?:
Your code essentially requires 2 things - A HashFuncConfig Object & a HashFuncGen

The Config Object: Compose this Object into your Hashtable.
This stores all the required attributes needed by the various different hash code, compress functions etc.
You can also add attributes to this when you create new hash codes or compression functions, its like a library.
Its a dataclass so most of the attributes can be overwritten or are generated automatically. (the only one thats required is table capacity for obvious reasons)
When Rehashing the table - you can recompute the attributes of the config class - with an inbuilt function called recompute(). 


The Hashfunction Generator:
This allows you to select and generate a hash code or compression function. In effect generating a compression function is the same as running both functions - so for brevitys sake can be ran alone.
If you create new hash code generators, or new compression functions, you will have to add them to this also. (and you might have to customize the logic.)
"""

@dataclass
class HashFuncConfig:
    """Configuration Class for HashFuncGen Strategy Class. -- Stores all the necessary attributes for the various methods to operate."""
    
    table_capacity: int # size of the hashtable capacity.

    # Salt for hash functions
    salt: bytes = field(init=False)
    salt_int: int = field(init=False)

    # PRF secret key
    prf_secret_key: bytes = field(init=False)

    # polynomial hash code
    polynomial_prime_weighting: int = 33 # small prime number: commonly 33, 37, 39, 41
    
    # cyclic shift hash code
    cyclic_bit_mask: BitMask = BitMask(2**64-1)  # This creates a 64-bit mask
    cyclic_shift_amount: int = 7    # used for cyclic shift hash code, shifts the bits by this much
    
    # MAD compress function
    mad_prime: int = field(init=False)  # used for MAD compression. 
    mad_scale: int = field(init=False)  # field is used to delay init until the attributes are computed
    mad_shift: int = field(init=False)
    
    # Universal Hashing
    universal_prime: int = field(init=False)
    universal_scale: int = field(init=False) # a must never be 0, Stretches and mixes the hash code before modulo.
    universal_shift: int = field(init=False)
  

    def __post_init__(self):
        """needed for computed attributes"""
        self._hash_utils = HashFuncUtils()
        self.recompute()
        
    def recompute(self, new_capacity: Optional[int] = None):
        """recalculates the MAD Compression attributes (prime, scale, shift - based on the new table capacity)"""
        if new_capacity is not None: self.table_capacity = new_capacity
        else: self.table_capacity = max(MIN_HASHTABLE_CAPACITY, self.table_capacity)

        # prf secret key:
        self.prf_secret_key: bytes = os.urandom(16)

        # salt for hash functions
        self.salt: bytes = os.urandom(16)  # new salt
        self.salt_int: int = int.from_bytes(self.salt, "big") # convert bytes salt to integer
        
        # MAD Compress Function - fixed after initialization (until table rehashing)
        self.mad_prime = self._hash_utils.find_next_prime_number(self.table_capacity)  # just slightly above table size.
        # must be smaller than prime attribute. (and cannot be a cofactor so cannot be 1)
        self.mad_scale = random.randint(2, self.mad_prime - 1)
        self.mad_shift = random.randint(2, self.mad_prime - 1)

        # Universal Hashing parameters -- # Re-randomize a,b only on resize, Use same a,b for all probes (critical for OA probing consistency)
        self.universal_prime = self._hash_utils.find_next_prime_number(self.table_capacity * 1000)
        self.universal_scale = random.randint(1, self.universal_prime - 1)  # a must never be 0
        self.universal_shift = random.randint(0, self.universal_prime - 1) 


class HashFuncGen():
    """
    Generates Hash Codes and Compression functions for Hash Tables
    Requires a Config Object - this is a dataclass that holds the attributes required for the succesful generation of the codes and functions.
    The Hash Code and Compress Function Inputs require ENUM TYPES - they can be found in the custom_types module
    """
    def __init__(self, key: iKey, config: 'HashFuncConfig', hash_code: HashCodeType = HashCodeType.CYCLIC_SHIFT, compress_func: CompressFuncType = CompressFuncType.MAD) -> None:
        self._config = config
        self._key = key.value   # this is part of the key class interface (iKey)
        self._hash_code = hash_code
        self._compress_func = compress_func

    def create_hash_code(self):
        """generate a hash code with the provided inputs"""
        if self._hash_code == HashCodeType.POLYNOMIAL:
            return HashCodesLib.polynomial_hash_code(self._key, self._config.polynomial_prime_weighting)
        elif self._hash_code == HashCodeType.CYCLIC_SHIFT:
            return HashCodesLib.cyclic_shift_hash_code(self._key, self._config.cyclic_shift_amount, self._config.cyclic_bit_mask)
        elif self._hash_code == HashCodeType.POLYCYCLIC:
            return HashCodesLib.cyclic_polynomial_combo_hash_code(self._key, self._config.cyclic_shift_amount, self._config.cyclic_bit_mask)
        elif self._hash_code == HashCodeType.SHA256:
            return HashCodesLib.sha_256_hash_code(self._key, self._config.salt)
        elif self._hash_code == HashCodeType.BLAKE2B:
            return HashCodesLib.keyed_prf_blake2b(self._config.prf_secret_key, self._key)
        else:
            raise KeyInvalidError("Error: Invalid Hash Code Type input. Check Enum Library for Valid Hash Code Types")

    def hash_function(self):
        """Generate an index value for a hash table (uses a hash code.) -- this is the compression function selector"""
        hash_code = self.create_hash_code()
        if self._compress_func == CompressFuncType.MAD:
            return CompressFunctionsLib.mad_compression_function(hash_code, self._config.mad_scale, self._config.mad_shift, self._config.mad_prime, self._config.table_capacity)
        elif self._compress_func == CompressFuncType.KMOD:
            return CompressFunctionsLib.k_mod_compression_function(hash_code,self._config.salt, self._config.table_capacity)
        elif self._compress_func == CompressFuncType.UNIVERSAL:
            return CompressFunctionsLib.universal_hashing_function(hash_code, self._config.universal_prime, self._config.universal_scale, self._config.universal_shift, self._config.table_capacity)
        elif self._compress_func == CompressFuncType.SHA256:
            return CompressFunctionsLib.sha_256_compress_function(hash_code, self._config.table_capacity)
        else:
            raise KeyInvalidError("Error: Invalid Hash Code Type input. Check Enum Library for Valid Hash Code Types")


# ------------------ Underlying Logic ---------------------
class HashFuncUtils:
    """General Utilities for Hash Functions to use."""
    @staticmethod
    def _is_prime_number(number: int):
        """Boolean Check if number is a prime."""
        if number < 2:
            return False
        for i in range(2, int(math.isqrt(number)) + 1):
            if number % i == 0:
                return False
        return True

    @staticmethod
    def find_next_prime_number(table_capacity: int):
        """Finds the next prime number larger than the current table capacity."""
        candidate = table_capacity + 1
        while True:
            if HashFuncUtils._is_prime_number(candidate):
                return candidate
            candidate += 1

    @staticmethod
    def convert_to_bytes(input) -> bytes:
        """
        Serialize common Python types to bytes deterministically.
        SHA-256 only accepts bytes. this function is necessary to guarantee derministic output for cryptographic hashing
        """
        if isinstance(input, bytes):
            return input
        if isinstance(input, str):
            return input.encode("utf-8")
        if isinstance(input, int):
            # big-endian variable-length integer serialization
            if input == 0:
                return b"\x00"
            length = (input.bit_length() + 7) // 8
            return input.to_bytes(length, "big", signed=False)
        # * for objects - with __hash__
        if hasattr(input, "__hash__"):
            return str(hash(input)).encode("utf-8")
        # fallback: use repr (deterministic for builtins)
        return repr(input).encode("utf-8")

    @staticmethod
    def convert_key_to_string(input):
        """Int, float, tuple will be converted to a string on entry. other types will be rejected (error)"""
        if isinstance(input, (int, float)):
            input = str(input)
        elif isinstance(input, tuple):
            string = f""
            for element in input:
                if isinstance(element, (int, float, str)):
                    element = str(element)
                    string += element
                else:
                    raise DsTypeError(f"Error: Cyclic Shift Hash Code requires tuple elements to be int, float or string!")
            input = string
        elif hasattr(input, "__hash__"):
            input = str(hash(input))    # this will use the __hash__ to create a hash of the object.
        else:
            raise DsTypeError(f"Error: While Converting a key to be used as a string in hash code functions, the element must be hashable or a valid datatype.")
        return input


class HashCodesLib:
    """
    Different types of hash codes for Hash Tables, they take a key and turn it into a long, unique integer, ready for processing by a compress function
    Value Based Hashing: For primitives and standard types (int, float, str, tuple etc...) 
    these will be equal if the contents are the same. E.g "apple" == "apple"
    Identity Based Hashing: For Objects and UDT. (Edge(), Vertex() etc.) 
    these will considered equal only by their id() - so only the same object will ever be considered equal.
    """
    def __init__(self) -> None:
        pass
    # -------------------------------- Hash Codes  --------------------------------
    @staticmethod
    def polynomial_hash_code(key, prime_weighting: int = 33):
        """polynomial hash code: uses Horners Method"""
        # * polynomial can only use strings.
        key = HashFuncUtils.convert_key_to_string(key)
        prime_weighting = prime_weighting  # small prime number: commonly 33, 37, 39, 41 - we will randomize and initialize on hashtable creation
        hash_code = 0
        # horner's method = hash * prime + char(ascii number)
        for character in key:
            hash_code = hash_code * prime_weighting + ord(character)
        return hash_code

    @staticmethod
    def cyclic_shift_hash_code(key, shift:int = 7, custom_bit_mask:Optional[int] = None):
        """Cyclic Shift Hash Code: uses bitwise shifting. Requires String key input."""
        # *  Cyclic shift can only use strings.
        key = HashFuncUtils.convert_key_to_string(key)

        word_bit_size = 64
        bit_mask = custom_bit_mask if custom_bit_mask else 2**64 - 1  # This creates a 64-bit mask
        hash_code = 0
        for char in key:
            # word_bit_size & bit_mask masks the result to 256 bits, effectively discarding any higher bits.
            hash_code = ((hash_code << shift) | (hash_code >> (word_bit_size - shift))) & bit_mask
            hash_code = hash_code ^ ord(char)
        return hash_code

    @staticmethod
    def cyclic_polynomial_combo_hash_code(key, shift: int = 7, custom_bit_mask:Optional[int] = None):
        """Combines Cyclic Shift and Polynomial techniques together to create a hash code."""
        key = HashFuncUtils.convert_key_to_string(key)
        prime_weighting = 33  # small prime number: commonly 33, 37, 39, 41 - we will randomize and initialize on hashtable creation
        bit_mask = custom_bit_mask if custom_bit_mask else 2**64 - 1  # This creates a 64-bit mask
        hash_code = 0
        # horner's method = hash * prime + char(ascii number)
        for character in key:
            hash_code = hash_code * prime_weighting + ord(character) & bit_mask
        # shifting bits
        hash_code ^= (hash_code << shift) & bit_mask
        hash_code ^= hash_code >> shift
        hash_code ^= hash_code << (shift // 2) & bit_mask
        return hash_code & bit_mask

    @staticmethod
    def sha_256_hash_code(key, salt):
        """Creates a Hash Code from SHA 256 algorithm"""
        # * 1. Convert key to bytes
        key_bytes = HashFuncUtils.convert_key_to_string(key).encode("utf-8")
        # * 2. SHA-256 digest
        digest = hashlib.sha256(key_bytes + salt).digest()
        # * 3. Convert digest to integer (hash code)
        hash_code = int.from_bytes(digest, "big")
        return hash_code

    @staticmethod
    def keyed_prf_blake2b(secret_key: bytes, key) -> int:
        """keyed PRF Hash Code generator. """
        key_bytes = HashFuncUtils.convert_to_bytes(key)
        digest = hashlib.blake2b(key_bytes, key=secret_key, digest_size=32)
        return int.from_bytes(digest.digest(), "big")


class CompressFunctionsLib:
    """Compression functions take a Hash Code, and convert it into a Hash Table Index, used to store key, value pairs"""
    def __init__(self) -> None:
        pass
    # -------------------------------- Compression Functions --------------------------------
    @staticmethod
    def k_mod_compression_function(hash_code, salt, table_capacity):
        """Takes a hash code and conforms it to the hash table size, and returns the index number"""
        # the division method: aka k-mod
        k_mod = (hash_code + salt) % table_capacity
        return k_mod

    @staticmethod
    def mad_compression_function(hash_code, scale, shift, prime, table_capacity):
        """The MAD Method - multiply - add - divide method: Takes a hashcode and conforms to table capacity - returns the index number"""
        # M-A-D Method core logic
        multiply = scale * hash_code
        add = multiply + shift
        divide = add % prime
        index = divide % table_capacity  # finally mod by table capacity
        return index

    @staticmethod
    def universal_hashing_function(hash_code, prime, scale, shift, table_capacity):
        """
        Universal Hashing Compression Function for use with Chaining Hash Tables, Probe-safe (as long as a,b stay fixed between resizes)
        for Optimal Distribution and High resilience to Adversarial DDOS attacks
        h(x) = ((a*x + b) mod p) mod m
        This is the strongest non-cryptographic hashing guarantee that exists (Carter–Wegman). (1 collision guaranteed per entry ever.)
        prime should be 1000 x your table size minimum.
        **Warning** this function is not probe safe!
        """
        # ! This Function is not safe for probing as is. needs to be modified. (in probe function library.)
        return ((scale * hash_code + shift) % prime) % table_capacity

    @staticmethod
    def sha_256_compress_function(hash_code, table_capacity):
        """Converts a SHA 256 Hash code into an index, with an added random salt to help protect against Hash Flood DOS attacks"""
        # ! requires a SHA-256 hash code to work correctly.
        return hash_code % table_capacity
