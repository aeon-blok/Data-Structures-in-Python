"""
Self Invention:

The Multi Node Access Linked List:
Expands on the head and tail node idea - derives a number of Access points or nodes in the linked list that operate to reduce traversal time.
these nodes operate as sector heads - halving the search time it takes to find a node.
You can now guarantee a maximum number of steps for any node: (distance between access nodes)/2

Jump Pointers:
Each node stores pointers to its grandparent and grandchild nodes. just write code for linking together...

# ! why isnt this done usually in linked lists? the answer is mostly historical, practical, and design philosophy.
Classic linked lists were designed in an era when memory was scarce -- Extra pointers per node could be prohibitive for large lists
Linked lists are not designed for random access: Most common use cases: sequential traversal, insertion, deletion -- For these, extra access nodes or shortcuts give little benefit -- If random access is needed, arrays are usually preferred

DLL + access nodes + pred/succ shortcuts
N = total number of nodes
k = number of access nodes (global jumps) (by making this a a logarithm or power of the total number of nodes we can achieve O(logN)) -- End result: deterministic O(log N) traversal, without randomization
s = stride of local pred/succ shortcuts (local jumps between nodes)

O(N/(k*s))
Each layer reduces steps multiplicatively
By tuning k and s, you can make traversal deterministically very fast, close to constant time.

Generalized Traversal Steps
Global jump: pick nearest access node → reduces search space from N → N/k ≈ N / logN
Hierarchical local jumps: halve remaining distance each time → log(N / logN) ≈ O(logN)
Result: deterministic O(logN) traversal

Recomputing access nodes over the whole list is O(N), but
With incremental or lazy strategies, the cost becomes negligible compared to traversal savings (recompute on doubling of linked list size)
Incremental Updates: Only update nearby access nodes when inserting/deleting Example: inserting near the head → maybe only first 1–2 access nodes need adjustment


that’s the main advantage of your approach:
Simplicity: no randomization, no multi-level probabilistic structure
Less bookkeeping: just a few deterministic access nodes and optional local shortcuts
Predictable traversal: worst-case is bounded, not probabilistic
DLL benefits: can traverse forward/backward, insert/delete easily

In contrast, a skip list:
Requires multiple levels of pointers per node
Needs randomization on insertion to maintain expected O(logN)
Traversal is probabilistic — worst-case could still be O(N) if unlucky
"""
