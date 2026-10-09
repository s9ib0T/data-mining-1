import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
import friends as fr
import preprocess as pp


# features

def observed_hours(bt):
    # hours outside class where the phone was on
    on = pp.presence(bt)
    on = on[~fr.in_class(on)]
    # SELECT user, COUNT(DISTINCT timestamp // 3600) FROM on GROUP BY user
    hours = on.assign(hour=on.timestamp // 3600).drop_duplicates(["hour", "user"])
    return hours.groupby("user").size()

def features(pairs, observed, ids):
    # one row per student in ids, from pairs outside class
    hour = pairs.timestamp // 3600
    # SELECT hour, user_a AS user, user_b AS other FROM pairs UNION SELECT hour, user_b, user_a FROM pairs
    near = pd.concat([
        pd.DataFrame({"hour": hour, "user": pairs.user_a, "other": pairs.user_b}),
        pd.DataFrame({"hour": hour, "user": pairs.user_b, "other": pairs.user_a})
    ]).drop_duplicates()
    # SELECT user, COUNT(DISTINCT hour) FROM near GROUP BY user
    company = near.drop_duplicates(["hour", "user"]).groupby("user").size()
    
    f = pd.DataFrame(index=ids)
    # share of observed hours with anyone near; observed counts -1/-2 scans too
    f["company"] = company.reindex(ids, fill_value=0) / observed.reindex(ids)
    # distinct students near during the 28 days
    # SELECT user, COUNT(DISTINCT other) FROM near GROUP BY user
    f["partners"] = near.groupby("user").other.nunique().reindex(ids, fill_value=0)
    return f


# anomaly detection

def scale(f, log=True):
    # log10 spreads out the low end, z-scores give both features the same weight
    X = np.log10(f) if log else f
    return pd.DataFrame(StandardScaler().fit_transform(X), index=f.index, columns=f.columns)

def lof(X, k=20):
    # sklearn stores the negative score -> flip it
    # about 1 is normal, higher is more unusual
    return pd.Series(-LocalOutlierFactor(n_neighbors=k).fit(X).negative_outlier_factor_, index=X.index)
