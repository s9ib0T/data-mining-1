from itertools import combinations
import matplotlib.pyplot as plt
import pandas as pd
from mlxtend.frequent_patterns import apriori
import preprocess as pp

WINDOW = 3600 # seconds per transaction (one hour)


# facebook

def friend_pairs(fb, users):
    # friendships inside users, frozenset to be able to be in a set
    f = fb[fb.user_a.isin(users) & fb.user_b.isin(users)]
    return set(map(frozenset, zip(f.user_a, f.user_b)))

