import sys

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

# 0/1 Knapsack Problem

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

    target_amt = 255
    coins = [1, 2, 5, 10, 20, 50]
    answer = min_coins(target_amt, coins)
    rec_answer = recursive_min_coins(target_amt, coins)
    print(f"Coin Change Problem: find the minimum number of coins from {coins} that can be used to make up the amount: {target_amt}?")
    print(f"Answer: The minimum number of coins needed is: {answer} and (recursive method) {rec_answer}")


if __name__ == "__main__":
    main()
