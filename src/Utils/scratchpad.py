import _collections_abc
from collections import Counter, deque

# just random testing..... delete later
# So basically, a deque is like a list, but optimized for front/back operations and with some extra “queue-friendly” methods.
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


# todo research priority queues
# todo positional list implementaiton
# todo study stacks and queues a bit more.
# todo refactor hash tables.
# todo refactor general tree
# todo implement __next__ for generators

# todo Solve problems: balanced parentheses, LRU cache, two-sum problem using hash tables
# todo trees -- Solve problems: heap sort, k largest elements, lowest common ancestor
# todo graphs -- Solve: connected components, shortest path, cycle detection
# todo Implement mini projects like a LRU cache, social network connections, or mini graph algorithms
