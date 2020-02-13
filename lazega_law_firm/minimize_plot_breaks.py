import numpy as np
from random import randint
import pickle
from utilities import num_communities
from time import time

N = 71
T = 3

stable_memberships = pickle.load(open("lazega_stable_K_2-4.p", "rb"))
stable_memberships = [tuple(np.array(membership[i * N:(i + 1) * N]) for i in range(T))
                      for membership in stable_memberships]

Ks = stable_memberships
K2s = Ks[:3]
K3s = Ks[3:5]
K4s = Ks[5:]

assert all(max(num_communities(s[0]), num_communities(s[1]), num_communities(s[2])) == 2 for s in K2s)
assert all(max(num_communities(s[0]), num_communities(s[1]), num_communities(s[2])) == 3 for s in K3s)
assert all(max(num_communities(s[0]), num_communities(s[1]), num_communities(s[2])) == 4 for s in K4s)


def num_breaks(sort, Ks):
    count = 0

    for m1, m2, m3 in Ks:
        m1, m2, m3 = m1[sort], m2[sort], m3[sort]
        for i in range(N - 1):
            if m1[i] != m1[i + 1]:
                count += 1
            if m2[i] != m2[i + 1]:
                count += 1
            if m3[i] != m3[i + 1]:
                count += 1
    return count


def one_iter(sort, Ks):
    current_breaks = num_breaks(sort, Ks)
    last_decrease = time()

    for iteration in range(10 ** 7):
        i, j = randint(0, N - 1), randint(0, N - 1)
        new_sort = sort.copy()
        new_sort[i], new_sort[j] = new_sort[j], new_sort[i]
        new_breaks = num_breaks(new_sort, Ks)

        if new_breaks < current_breaks:
            current_breaks = new_breaks
            sort = new_sort
            last_decrease = time()
            print("strict decrease to", current_breaks)
        elif new_breaks == current_breaks:
            current_breaks = new_breaks
            sort = new_sort

        if time() - last_decrease > 30:
            break

    return current_breaks, sort


sort2 = np.array([46, 21, 20, 0, 26, 23, 19, 37, 35, 22, 7, 42, 25, 38, 39, 12, 10,
                  56, 67, 51, 55, 66, 70, 64, 40, 48, 53, 65, 54, 68, 61, 44, 18, 69,
                  63, 60, 43, 36, 59, 41, 52, 8, 14, 3, 28, 33, 1, 9, 47, 11, 16,
                  15, 24, 29, 13, 32, 30, 6, 34, 58, 49, 57, 62, 50, 45, 5, 2, 4,
                  17, 31, 27])
sort3 = np.array([35, 61, 56, 55, 66, 68, 53, 54, 67, 70, 39, 38, 40, 64, 51, 42, 65,
                  48, 20, 23, 7, 37, 22, 0, 25, 19, 26, 10, 12, 21, 16, 36, 46, 18,
                  43, 11, 1, 14, 3, 15, 52, 9, 28, 8, 33, 60, 69, 45, 41, 63, 44,
                  59, 47, 24, 17, 27, 62, 50, 32, 29, 30, 6, 34, 13, 57, 2, 49, 31,
                  5, 58, 4])
sort4 = np.array([19, 0, 7, 35, 42, 38, 39, 37, 53, 66, 64, 40, 67, 54, 70, 68, 48,
                  56, 65, 21, 12, 10, 23, 26, 22, 25, 20, 46, 51, 55, 61, 69, 60, 63,
                  41, 3, 1, 14, 9, 11, 8, 16, 15, 28, 18, 52, 36, 43, 59, 44, 45,
                  33, 47, 17, 4, 27, 5, 62, 2, 30, 32, 29, 50, 49, 24, 13, 58, 6,
                  57, 31, 34])

sort_original = np.array([21, 46, 53, 68, 55, 70, 56, 65, 40, 48, 54, 67, 66, 51, 64, 38, 39,
                          42, 37, 20, 23, 25, 26, 12, 22, 10, 0, 7, 19, 35, 61, 69, 63, 60,
                          41, 15, 11, 8, 9, 16, 3, 1, 14, 36, 43, 52, 18, 28, 47, 33, 59,
                          44, 45, 62, 17, 2, 30, 34, 5, 6, 50, 58, 57, 32, 31, 49, 4, 27,
                          13, 29, 24])

sort = np.array([46, 21, 64, 55, 48, 54, 68, 40, 70, 56, 65, 66, 53, 51, 67, 38, 42,
                 39, 37, 26, 20, 10, 12, 22, 25, 0, 23, 19, 7, 35, 61, 69, 63, 41,
                 28, 16, 15, 8, 11, 9, 14, 1, 3, 36, 18, 52, 43, 47, 33, 44, 60,
                 59, 45, 31, 34, 27, 62, 5, 49, 30, 58, 50, 17, 57, 4, 32, 6, 2,
                 24, 13, 29])
# sort = np.unravel_index(np.argsort(membership1, axis=None), membership1.shape)

print(one_iter(sort, Ks))
