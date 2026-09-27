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


# together

def in_class(prox):
    # weekdays 08:00-17:00, the "busy" hours in activity plot
    hour = (prox.timestamp % pp.DAY) // 3600
    day = prox.timestamp // pp.DAY
    weekend = (day % 7).isin([0, 6])
    return ~weekend & hour.between(8, 16) # covers 08:00-16:59

def together(prox, rssi, teaching=False):
    # pairs at or above the rssi threshold, inside or outside teaching hours
    keep = (prox.rssi >= rssi) & (in_class(prox) == teaching)
    return prox[keep].reset_index(drop=True)

