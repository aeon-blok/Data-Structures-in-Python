import _collections_abc
from collections import Counter, deque
import heapq
# just random testing..... delete later

# slicing not supported in python deque - we can fix that :D
dq = deque([2,3,5,6,7,7,8,0])
print(dq)
dq.append(25)
dq.appendleft(1)
print(dq)
dq.pop()
dq.popleft()
print(dq)
dq.rotate(4)
print(dq)
dq.rotate(-4)
print(dq)
print(dq.count(7))
dq.remove(7)
print(dq)
dq.extend([100,200,300])
dq.extendleft([1000,2000,3000])
print(dq)
dq.clear()
print(dq)

# heapq module test - its shite
newlist = [i for i in range(25)]
heapq.heapify(newlist)
print(newlist)
heapq.heappush(newlist,25)
print(newlist)
popped = heapq.heappushpop(newlist, 100)
print(newlist)
popo = heapq.heapreplace(newlist, 1000)
print(newlist)


# todo refactor tree nodes to use inheritance.
# todo refactor hash tables.
# todo implement __next__ for generators

# todo Solve problems: balanced parentheses, LRU cache, two-sum problem using hash tables
# todo trees -- Solve problems: heap sort, k largest elements, lowest common ancestor
# todo graphs -- Solve: connected components, shortest path, cycle detection
# todo Implement mini projects like a LRU cache, social network connections, or mini graph algorithms
# todo need to add recursion guards to the representations to prevent recursive loops with __str__....