from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
RAW = ROOT / "data" / "copenhagen"
PROCESSED = ROOT / "data" / "processed"

BIN = 300 # seconds between bluetooth scans
N_BINS = 8064 # 28 days of 5 minute bins
DAY = 86400 # seconds
EMPTY = -1 # user_b of an empty scan
OUTSIDE = -2 # user_b of a device outside the study

# default size for single plots
plt.rcParams["figure.figsize"] = (6, 3.5)


# loading data

def load_bt():
    # header line starts with "# " so skip it and name the cols
    return pd.read_csv(
        RAW / "bt_symmetric.csv",
        skiprows=1,
        names=["timestamp", "user_a", "user_b", "rssi"],
        dtype={"timestamp": "int32", "user_a": "int16", "user_b": "int16", "rssi": "int8"}
    )
    
def load_fb():
    return pd.read_csv(
        RAW / "fb_friends.csv",
        skiprows=1, names=["user_a", "user_b"], dtype="int16"
    )


def load_calls():
    return pd.read_csv(
        RAW / "calls.csv",
        dtype={"timestamp": "int32", "caller": "int16", "callee": "int16", "duration": "int32"},
    )


def load_sms():
    return pd.read_csv(
        RAW / "sms.csv",
        dtype={"timestamp": "int32", "sender": "int16", "recipient": "int16"}
    )


# checks

def presence(bt):
    # one row per (bin, student) where the phone was on
    # own row of any kind, or seen by another student
    own = bt[["timestamp", "user_a"]].rename(columns={"user_a": "user"})
    seen = bt.loc[bt.user_b >= 0, ["timestamp", "user_b"]].rename(columns={"user_b": "user"})
    return pd.concat([own, seen]).drop_duplicates()


# exploration plots

def plot_row_types(bt):
    shares = pd.Series(
        {
            "real pair": (bt.user_b >= 0).mean(),
            "empty scan": (bt.user_b == EMPTY).mean(),
            "outside device": (bt.user_b == OUTSIDE).mean(),
        }
    )
    fig, ax = plt.subplots()
    ax.bar(shares.index, shares)
    ax.set_ylabel("share of rows")
    ax.set_title("Bluetooth rows by type")
    return fig


def plot_rssi(bt):
    real = bt.rssi[bt.user_b >= 0]
    outside = bt.rssi[bt.user_b == OUTSIDE]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    bins = np.arange(-108, 22)
    ax1.hist(real, bins=bins, histtype="step", label="real pair")
    ax1.hist(outside, bins=bins, histtype="step", label="outside device")
    ax1.set_xlabel("RSSI (dBm)")
    ax1.set_ylabel("rows")
    ax1.set_title("RSSI distribution")
    ax1.legend()
    ax2.boxplot([real, outside], tick_labels=["real pair", "outside device"], orientation="horizontal")
    ax2.set_xlabel("RSSI (dBm)")
    ax2.set_title("RSSI box plot")
    return fig


def plot_activity(real):
    hour = (real.timestamp % DAY) // 3600
    day = real.timestamp // DAY
    by_hour = hour.value_counts().sort_index()
    by_day = day.value_counts().sort_index()
    # day 0 is a sunday, so days 0 and 6 of each week are weekend
    weekend = by_day[(by_day.index % 7).isin([0, 6])]
    weekday = by_day.drop(weekend.index)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.bar(by_hour.index, by_hour.values / 1e3)
    ax1.set_xlabel("hour of day")
    ax1.set_ylabel("real-pair rows (thousands)")
    ax1.set_title("Activity by hour")
    ax2.bar(weekday.index, weekday.values / 1e3, label="weekday")
    ax2.bar(weekend.index, weekend.values / 1e3, label="weekend")
    ax2.set_xlabel("day of study")
    ax2.set_title("Activity by day")
    ax2.legend()
    return fig


def plot_partners(real):
    pairs = real[["user_a", "user_b"]].drop_duplicates()
    # each pair counts once for both students
    n = pd.concat([pairs.user_a, pairs.user_b]).value_counts()
    fig, ax = plt.subplots()
    ax.hist(n, bins=50)
    ax.set_xlabel("distinct partners over 28 days")
    ax.set_ylabel("students")
    ax.set_title("Partners per student")
    return fig


def plot_pair_bins(real):
    # bins per pair
    n = real.groupby(["user_a", "user_b"]).size()
    fig, ax = plt.subplots()
    # log-spaced edges rounded to whole bins, so small counts get no empty gaps
    edges = np.unique(np.geomspace(1, n.max() + 1, 40).astype(int))
    ax.hist(n, bins=edges)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("bins together (log scale)")
    ax.set_ylabel("pairs (log scale)")
    ax.set_title("Time together per pair")
    return fig


# cleaning

def clean_bt(bt):
    # keep real pairs, drop the 4 rows with rssi +20
    return bt[(bt.user_b >= 0) & (bt.rssi < 0)].reset_index(drop=True)

def clean_fb(fb):
    # drop self-loops
    return fb[fb.user_a != fb.user_b].reset_index(drop=True)

def clean_sms(sms):
    # drop duplicates
    return sms.drop_duplicates().reset_index(drop=True)


# student table

def students(bt, fb, calls, sms):
    # bt is raw, since coverage needs the rows without a pair
    # fb and sms are cleaned
    real = bt[bt.user_b >= 0]
    in_bt = set().union(real.user_a, real.user_b)
    in_fb = set().union(fb.user_a, fb.user_b)
    in_comm = set().union(calls.caller, calls.callee, sms.sender, sms.recipient)
    # every id in any file, also students with only empty scans
    ids = pd.Index(sorted(set().union(bt.user_a, in_bt, in_fb, in_comm)), name="user")

    # counts per student of 5 minute time slots (bins) where:
    # phone on: own row of any kind, or seen by another student
    n_on = presence(bt).groupby("user").size()
    # own phone scanned
    n_own = bt.groupby("user_a").timestamp.nunique()
    # own scan found an outside device
    n_outside = bt[bt.user_b == OUTSIDE].groupby("user_a").timestamp.nunique()

    st = pd.DataFrame(index=ids)
    st["coverage"] = n_on.reindex(ids, fill_value=0) / N_BINS
    # 0 when no outside device was seen, empty when the student has no own rows
    st["outside_share"] = n_outside.reindex(n_own.index, fill_value=0) / n_own
    # divide by n_own, since only own scans can find an outside device
    st["in_bt"] = ids.isin(in_bt)
    st["in_fb"] = ids.isin(in_fb)
    st["in_comm"] = ids.isin(in_comm)
    return st

def plot_coverage(st):
    fig, ax = plt.subplots()
    ax.hist(st.coverage[st.coverage > 0], bins=40)
    ax.set_xlabel("coverage (share of 5 minute bins)")
    ax.set_ylabel("students")
    ax.set_title("Coverage per student")
    return fig