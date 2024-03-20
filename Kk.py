def bin_search(a, target):
    low = 0
    high = len(a) - 1
    while low <= high:
        mid = (low + high) // 2
        midVal = a[mid]
        if midVal < target:
            low = mid + 1
        elif midVal > target:
            high = mid - 1
        else:
            return mid
    return -1


a = [1, 2, 3, 4, 5, 6, 7, 8]
target = 6
found_index = -1
found_index = bin_search(a, target)
if found_index == -1:
    print("not found")
else:
    print('%s:%d' % ('found', found_index))

a = [3, 5, 7, 9, 11, 13, 17, 23, 29]
target = 2
found_index = -1
found_index = bin_search(a, target)
if found_index == -1:
    print('not found')
else:
    print('%s:%d' % ('found', found_index))
