import _collections_abc
from collections import Counter, deque
from _collections_abc import MutableSet
import heapq
from ctypes import string_at
from sys import getsizeof
from binascii import hexlify
import enum
from enum import Flag, auto
from pprint import pprint




# creates a 11x11 Zero Matrix
dp = []
for i in range(10+1):
    row = []
    for j in range(10+1):
        row.append(0)
    dp.append(row)

pprint(dp)




















# # The underscores _ are just for readability, they are ignored by Python.
# a = 0b01010000_01000001_01010100
# print(a)

# # prints the raw memory contents of the Python integer object as hex.
# print(hexlify(string_at(id(a), getsizeof(a))))
# # id(a) - Returns the memory address of the object a (in CPython, this is the pointer to the object).


# # testing flag module for enums
# class EnumTest(Flag):
#     CAR = auto()
#     PLANE = auto()
#     BOAT = auto()

# # can input any of the type choices
# new_test_type = EnumTest.CAR | EnumTest.PLANE


# def testfunc(type_input):
#     """type narrowing test function: Must include boat class."""
#     try:
#         if EnumTest.BOAT not in type_input:
#             raise Exception("Does not include the critical boat!")
#         else:
#             return f"Correct Type! {type_input}"
#     except Exception as e:
#         print(f"{e}")

# print(testfunc(new_test_type))  # Output: Does not include the critical boat!



# # just random testing..... delete later

# # slicing not supported in python deque - we can fix that :D
# dq = deque([2,3,5,6,7,7,8,0])
# print(dq)
# dq.append(25)
# dq.appendleft(1)
# print(dq)
# dq.pop()
# dq.popleft()
# print(dq)
# dq.rotate(4)
# print(dq)
# dq.rotate(-4)
# print(dq)
# print(dq.count(7))
# dq.remove(7)
# print(dq)
# dq.extend([100,200,300])
# dq.extendleft([1000,2000,3000])
# print(dq)
# dq.clear()
# print(dq)

# # heapq module test - its shite
# newlist = [i for i in range(25)]
# heapq.heapify(newlist)
# print(newlist)
# heapq.heappush(newlist,25)
# print(newlist)
# popped = heapq.heappushpop(newlist, 100)
# print(newlist)
# popo = heapq.heapreplace(newlist, 1000)
# print(newlist)


# todo implement __next__ for generators
# todo update binary tree 2 child deletion (also avl and rb trees) to actually swap the node and not just the contents.
# todo Solve problems: balanced parentheses, LRU cache, two-sum problem using hash tables
# todo trees -- Solve problems: heap sort, k largest elements, lowest common ancestor
# todo graphs -- Solve: connected components, shortest path, cycle detection
# todo Implement mini projects like a LRU cache, social network connections, or mini graph algorithms

