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


# transactions

def transactions(pairs, window=WINDOW):
    # one basket per student and window, with the student and everyone near them
    # student in own basket, else pair (1, 2) never gives itemset {1, 2}
    # example below: window 0 with pairs (1, 2) and (1, 3)

    w = pairs.timestamp // window
    a, b = pairs.user_a, pairs.user_b
    
    # each pair gives 4 rows, a and b both go in a's and b's basket
    # user = basket owner, item = student in the basket
    #   w  user  item
    #   0     1     1   <- from (1, 2)
    #   0     1     2
    #   0     2     1
    #   0     2     2
    #   0     1     1   <- from (1, 3), same row again
    #   0     1     3
    #   0     3     1
    #   0     3     3
    
    rows = pd.concat([
        pd.DataFrame({"w": w, "user": a, "item": a}),
        pd.DataFrame({"w": w, "user": a, "item": b}),
        pd.DataFrame({"w": w, "user": b, "item": a}),
        pd.DataFrame({"w": w, "user": b, "item": b}),
    ]).reset_index(drop=True)

    # crosstab counts rows per basket (w, user) and item
    #   item    1  2  3
    #   w user
    #   0 1     2  1  1
    #     2     1  1  0
    #     3     1  0  1
    # > 0 turns counts into true/false, the input fpgrowth and apriori want
    #   item       1      2      3
    #   w user
    #   0 1     True   True   True
    #     2     True   True  False
    #     3     True  False   True
    
    return pd.crosstab([rows.w, rows.user], rows.item) > 0


# mining

def mine(X, min_count):
    # min_count: transactions a group must appear in
    # low_memory checks candidates in parts, the default needs 37 GB with teaching hours
    return apriori(
        X,
        min_support=min_count / len(X),
        use_colnames=True, 
        low_memory=True
    )

def maximal(fi):
    # groups of 2 or more with no frequent superset
    groups = fi[fi.itemsets.apply(len) >= 2]
    sets = list(groups.itemsets)
    keep = [not any(s != t and s.issubset(t) for t in sets) for s in sets]
    return groups[keep]


# results

def score(fi, friends, users):
    # frequent pairs where both students have facebook data
    # every pair inside a frequent group is frequent too, so this covers all groups

    pairs = {s for s in fi.itemsets if len(s) == 2 and s.issubset(users)}
    hits = sum(p in friends for p in pairs)
    n = len(users)
    possible = n * (n - 1) / 2
    share = hits / len(pairs)
    return pd.Series({
        "pairs": len(pairs),
        "friend share": share,
        "times baseline": share / (len(friends) / possible),
        "friends found": hits / len(friends),
    })

def group_share(group, friends):
    # share of pairs in the group that are facebook friends
    
    pairs = [frozenset(p) for p in combinations(group, 2)]
    return sum(p in friends for p in pairs) / len(pairs)

def plot_shares(shares):
    fig, ax = plt.subplots()
    ax.barh(shares.index, shares)
    ax.bar_label(ax.containers[0], labels=[f"{s:.0%}" for s in shares], padding=3)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("share of pairs that are facebook friends")
    ax.set_title("Friendship by kind of proximity")
    return fig