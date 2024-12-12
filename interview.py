# list of ints, and a target (also an int)
# return boolean, if list containst 2 values that add up to target, false otherwise.

def check_if_vals_add_up_to_target(arr, target):
    # Time: O(n^2) - 2 for loops
    # Time: O(n): set
    diff = set([target - e for e in arr])

    # [99, 100, 101, 102]
    # target - e = diff
    # e + diff = target

    for e in arr:
        if e in diff:
            return True
    return False


if __name__ == "__main__":
    arr = [1, 2, 3, 4]
    target = 100
    print(check_if_vals_add_up_to_target(arr, target))
