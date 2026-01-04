import sys
from pprint import pprint
import time
import faker
import random

# region rod cutting
"""
rod cutting problem
rod length = 8
length:    1  2  3  4  5  6  7  8
price    : 1  5  8  9 10 17 17 20

task:
cut the rod into any number of pieces
get the max value for the rod length given.

constraints:
Unlimited cuts allowed.
Order of pieces does not matter.
"""

def rod_cutting_memo(rod_length: int, lengths: list[int], prices: list[int], memo=None):
    """
    rod cutting problem - follows CLRS guide.
    Top Down Memoization approach -- O(N2)
    The Key is to find the length with the highest value - and utilize it as much as possible.

    """

    # State = rod length
    # * create memo cache - needs to be 1 bigger than the rod length to account for all sublengths.
    if memo is None:
        memo = [-sys.maxsize] * (rod_length + 1)

    # * base case
    if rod_length == 0:
        return 0
    
    # * return cache
    if memo[rod_length] != -sys.maxsize:
        return memo[rod_length]
    
    # init as sentinel
    profit = -sys.maxsize
    # loop through the prices array.
    for idx, cut in enumerate(lengths):
        # ensure cut is less than the total rod length.
        if cut <= rod_length:
            # get the max between the current max and reductive step.
            profit = max(profit, prices[idx] + rod_cutting_memo(rod_length-cut,lengths, prices, memo))

    # * cache result: after the profit for this path has been finalized - add to cache.
    memo[rod_length] = profit

    # after recursive steps finish return the final profit.
    return profit

def rod_cutting_tabulation(rod_length:int, lengths:list[int], prices: list[int]):
    """
    Bottom up (tabulation) solution to classic rod cutting problem
    time complexity is O(NK)
    """

    # stores the maximum profit for a rod of a specific length (the array index)
    table = [-sys.maxsize] * (rod_length + 1)
    table[0] = 0    # edge case

    # iterate over every length in the range from 1 to rod length
    for length in range(1, rod_length+1):
        profit = -sys.maxsize

        # iterates over the specific length units in the length array.
        # the index is used to retrieve the matching price to the length.
        for idx, cut in enumerate(lengths):
            # size safety check
            if cut <= rod_length:
                # * reductive step
                profit = max(profit, prices[idx] + table[length-cut])

        # cache result to table array.
        table[length] = profit
    # return result for target rod length
    return table[rod_length]
# endregion

# region fibonacci
# Find fibonacci number for specified index
def dp_td_find_fib_number(index: int, memo=None):
    """
    A Top Down (Memoization) Approach to calculating the fibonacci number that corresponds to the specified index.
    In this definition 0 - is not counted as a fib number.
    """

    if memo is None:
        memo = {}  # dict not set

    # * base case
    if index <= 1:
        return index

    # recurrence relation: (state transition) n = n-1 + n-2
    # n is reduced via every recursive call - this is the reductive step towards the base case.
    if index not in memo:
        memo[index] = dp_td_find_fib_number(index-1, memo) + dp_td_find_fib_number(index-2, memo)

    # exit condition - after base case is met.
    return memo[index]

def dp_bu_find_fib_number(index: int):
    """A Bottom up (Tabulation) approach. calculates the fib number for the specified index."""

    # exit condition - already have answer. (similar to base case)
    if index <= 1:
        return index
    
    table = [0] * (index + 1)
    table[1] = 1    # first entry into table

    # reductive step
    for i in range(2, index+1):
        table[i] = table[i-1] + table[i-2]
    # exit condition - after loop finishs
    return table[index]

def dp_bu_find_fib_number_space_optimized(index: int):
    """a Bottom Up (Tabulation) approach - space optimized, now Space Complexity is O(1) instead of O(N)"""

    # exit condition
    if index <= 1:
        return index

    a, b = 0, 1

    # reductive step towards base case. (starts from 2 because exit condition from 1 or 0 already exists.)
    for _ in range(2, index+1):
        a, b = b, a + b

    # exit condition - after iterative loop finishes.
    return b
# endregion

# region coin change
# 1D Coin Change Problem
def min_coins(amt: int, coins: list):
    """
    A bottom up (tabulation) approach for solving the classic Coin Change Problem

    Task:
        You are given: an array of coin denominations (positive integers) and a target amount A
        Determine the minimum number of coins needed to make up amount A.
    rules: 
        You may use unlimited coins of each denomination. If it’s not possible, return -1.
    example: 
        amt=6, coins=[1,3,4]

    the approach take has a Time Complexity of O(amt x coins) and Space Complexity of O(amt) (for the lookup table)
    """

    # edge case:
    if amt < 0 or len(coins) <= 0:
        return -1
    
    # the state is the amt

    # * initiaize lookup table (array) 
    # uses a proxy number that will never be reached.
    min_coins = [sys.maxsize] * (amt+1)
    min_coins[0] = 0

    # * iterate through amounts starting from 1 to the target
    for i in range(1, amt + 1):
        # iterate through the coins
        for coin in coins:
            # size check (is coin smaller than the target amount) -- this line prevents negative indices
            if coin <= i:
                # * State Transition / recurrence relation:
                # Test: choose a coin - and compare to the target amt. observe the remaining amt [i-coin].
                # Retrieval: We already have this result cached. collect it. 
                # Increment: Now we increment an additional coin (+1) to this because our test coin has been substracted from the target amt.
                # Compare: Finally we compare this new total (Min_Number_of_Coins) to the already cached total - to see which one is lower: min(cache, new_result)
                min_coins[i] = min(min_coins[i], min_coins[i-coin] + 1)

    # final result - if it was not possible to use any coins to form the amt, return -1
    return min_coins[amt] if min_coins[amt] != sys.maxsize else -1

def recursive_min_coins(amt: int, coins: list, memo=None):
    """recursive top down (memoization) version of coin change solution"""

    if memo is None:
        memo = {} # dict

    def _recurse_deeper(amt):
        """recursive helper function"""
        # * base case: target amount hit.
        if amt == 0: return 0

        # * base case: target amount NOT hit
        if amt < 0: return sys.maxsize

        # * check cache and return result.
        if amt in memo:
            return memo[amt]

        # * State Transition / Recurrence relation:
        memo[amt] = min(_recurse_deeper(amt - coin) + 1 for coin in coins)
        return memo[amt]

    result = _recurse_deeper(amt)
    return result if result != sys.maxsize else -1
# endregion

# region 0/1 Knapsack
# 0/1 Knapsack Problem
# ! You have a knapsack with capacity=10kg
# ! you have 5 items = {A: (2kg, $3), B: (3kg, $4), C: (4kg, $8), D: (5kg, $8), E: (9kg, $10)}
# Aim is to get the maximum value combination of items in the knapsack.
# we need to record the items that were selected. you cannot use an item more than once.
# we only need to record 1 combination that gets the max value, if there are multiple possible solutions

# C+D = 9kg, $16 is the optimal combo for the example

def knapsack_naive_recursion(capacity, item_names, weights, prices):
    """
    A recursive solution to the 0/1 Knapsack problem.
    Time Complexity -- O(2n) Exponential time. This is very slow
    Every valid combination of items is considered, so the maximum total value will always be found. (unlike greedy algo)
    Generally 0/1 Knapsack is solved utilizing a 2D matrix. not in this case.
    We iterate through each item - with a binary choice - (take item, or skip)
    """

    def recursive_step(idx, space):
        """helper method that recursively steps through each item and either adds it to the knapsack or skips it."""

        # * base case (all items checked, or zero space in knapsack.)
        if idx == len(weights) or space == 0:
            return 0, []

        # * take current item:
        # init variables
        take_profit, take_items = 0, []
        weight = weights[idx]
        price = prices[idx]
        name = item_names[idx]

        # only runs if the item weight fits in the knapsack. (else we skip the current item)
        if weight <= space:
            take_profit, take_items = recursive_step(idx + 1, space - weight)
            # increment profit - this is optimal value / max value we need
            take_profit += price
            # increment take items list - this is the combination of items that make up the max value.
            take_items += [name]

        # * skip current item: - its too large to fit in knapsack
        skip_profit, skip_items = recursive_step(idx + 1, space)

        # * compare the two branches: return the best results
        if take_profit > skip_profit:
            return take_profit, take_items
        else:
            return skip_profit, skip_items

    return recursive_step(0, capacity)

def knapsack_memo(capacity, items):
    """
    Solves the 0/1 Knapsack problem via top down recursion with memoization cache. 
    returns both the optimal value and the combination of items chosen.
    """
    num_items = len(items)
    sentinel = -sys.maxsize
    # * initialize memo cache
    # table[item][capacity_weight] = max profit for capacity weight (including all items up to currently selected item)
    table = [[sentinel] * (capacity + 1) for _ in range(num_items + 1)]

    def recurse(item_idx, current_capacity):
        """
        Helper recursive function - returns the optimal max value
        will fill the memo matrix with all the max values achievable at every capacity size.
        """
        # * base case:
        if item_idx == 0 or current_capacity == 0:
            return 0
        
        # * check memo cache
        if table[item_idx][current_capacity] != sentinel:
            return table[item_idx][current_capacity]
        
        # * unpack tuple: (-1 accounts for the 0 based vs 1 based indexing discrepancy)
        _, weight, value = items[item_idx-1]

        # Descision Time:
        # * Skip Item - doesnt fit in knapsack
        if weight > current_capacity:
            # recurse into a smaller sized item and run the same descision tree
            table[item_idx][current_capacity] = recurse(item_idx-1, current_capacity)
        # * take or skip depending on the max value received - fits in the knapsack
        else:
            take = recurse(item_idx-1, current_capacity-weight) + value
            skip = recurse(item_idx-1, current_capacity)
            # * update memo cache with the greater max value from either path.
            table[item_idx][current_capacity] = max(skip, take)
        
        return table[item_idx][current_capacity]
    
    # starts at max capacity and works backwards - computes the optimal value
    optimal_value = recurse(num_items, capacity)

    # * Backtracking - collecting the list of items that amount to the optimal max value.
    choices = []
    item = num_items
    current_weight = capacity
    # iterate through items in the table - starting from max capacity and decreasing (both capacity and item) every time an item is chosen.
    while item > 0 and current_weight > 0:
        # unpack tuple and append to list for return to sender
        name, weight, value = items[item-1]
        # value is different from the entry in the column above, then this item was taken.
        if weight <= current_weight and table[item][current_weight] != table[item-1][current_weight]:
            choices.append(name)
            # decrement weight tracker to move back through memo cache.
            current_weight -= weight
        # decrement item tracker to move to the next item.
        item -= 1

    # preserves the original insertion order
    choices.reverse()

    return optimal_value, choices


# endregion

# -------------------------------- Main: Client Facing Code: --------------------------------
def main():
    # # 0 based indexing
    # index = 9
    # print(f"What is the {index}th fibonacci number? result={dp_td_find_fib_number(index)}")
    # print(f"What is the {index}th fibonacci number? result={dp_bu_find_fib_number(index)}")
    # print(f"What is the {index}th fibonacci number? result={dp_bu_find_fib_number_space_optimized(index)}")

    # # 1 based indexing
    # new_index = 25
    # cache = []
    # for i in range(1, new_index+1):
    #     cache.append(dp_bu_find_fib_number_space_optimized(i))
    # print(f"Fib numbers from 1-{new_index}: {cache}")

    # target_amt = 255
    # coins = [1, 2, 5, 10, 100, 25]
    # answer = min_coins(target_amt, coins)
    # rec_answer = recursive_min_coins(target_amt, coins)
    # print(f"Coin Change Problem: find the minimum number of coins from {coins} that can be used to make up the amount: {target_amt}?")
    # print(f"Answer: The minimum number of coins needed is: {answer} and (recursive method) {rec_answer}")

    # print(f"Rod Cutting Problem:")
    # rod_length_prices = [1,5,8,9,10,17,17,20]
    # rod_lengths = [1,2,3,4,5,6,7,8]
    # target_rod_size = 3054
    # print(f"Rod Cutting Problem: Cut the Target rod {target_rod_size}m into any number of pieces.")
    # print(f"Our Task is to get the most profit from our rod")
    # print(f"The prices for each specific length are \n{rod_lengths}\n{rod_length_prices}")
    # print(f"Constraints: The problem is unbounded, you can use unlimited cuts of the same length (e.g. 1+1+1+1)")
    # print(f"The order of the pieces dont matter...")
    # sys.setrecursionlimit(10000)
    # print(f"The Optimal Profit we can get is: ${rod_cutting_memo(target_rod_size,rod_lengths, rod_length_prices)} and ${rod_cutting_tabulation(target_rod_size, rod_lengths, rod_length_prices)}")

    fake = faker.Faker()
    fake.seed_instance(642)

    print(f"0/1 Knapsack Problem:")
    knapsack_size = 50
    item_amount = 100
    item_names = [fake.word() for _ in range(item_amount)]

    item_weights = [random.randint(1, 50) for i in range(item_amount)]
    item_prices = [random.randint(1, 100) for i in range(item_amount)]
    items_tuple = [(item_names[i], item_weights[i], item_prices[i]) for i,_ in enumerate(item_weights)]

    print(f"We have a knapsack that has a carrying capacity of {knapsack_size}kg")
    print(f"We have a collection of items, with associated prices. \n{item_weights}\n{item_prices}")
    print(f"Aim to put a combination of items in the knapsack which has the highest price value.")
    print(f"Each item can only be added once.")

    naive_start = time.perf_counter()
    profit, combos = knapsack_naive_recursion(knapsack_size, item_names, item_weights, item_prices)
    naive_end = time.perf_counter()
    naive_time = f"{naive_end-naive_start:03f} Secs"

    memo_start = time.perf_counter()
    value, choices = knapsack_memo(knapsack_size, items_tuple)
    memo_end = time.perf_counter()
    memo_time = f"{memo_end-memo_start:03f} Secs"

    print(f"Naive Recursion Answer: ${profit} with items: {combos} in time: {naive_time}")
    print(f"DP Top Down Memoization Answer: ${value} with items: {choices} in time: {memo_time}")


if __name__ == "__main__":
    main()
