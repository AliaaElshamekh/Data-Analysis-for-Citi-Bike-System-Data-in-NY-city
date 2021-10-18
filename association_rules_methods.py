import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
import json
import math
from collections import OrderedDict
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv("201910-citibike-tripdata.csv")
df.head()
df.info()
print (df.shape)
df.isnull().sum()
df = df.dropna(axis=1, how='any')
df.shape


df_loc = pd.read_csv("station_information.csv")
df_loc.head()

df_flow = pd.read_csv("station_flow.csv")
df_flow['time'] = pd.to_datetime(df_flow['time'], format='%Y-%m-%d %H:%M:%S')
df_flow.head()

from apyori import apriori


def apriori_find_association_rules(dataset, minsup, minconf):
    records = list(apriori(dataset, min_support=minsup, min_confidence=minconf))
    return records


def apriori_show_mining_results(records):
    ap = []
    for record in records:
        converted_record = record._replace(ordered_statistics=[x._asdict() for x in record.ordered_statistics])
        ap.append(converted_record._asdict())

    # print("Frequent Itemsets:\n------------------")
    # for ptn in ap:
    #    print('({})  support = {}'.format(", ".join(ptn["items"]), round(ptn["support"], 3)))
    # print()
    print("Rules:\n------")
    for ptn in ap:
        for rule in ptn["ordered_statistics"]:
            head = rule["items_base"]
            tail = rule["items_add"]
            if len(head) == 0 or len(tail) == 0:
                continue
            confidence = rule["confidence"]
            print('({}) ==> ({})  confidence = {}'.format(', '.join(head), ', '.join(tail), round(confidence, 3)))
    print()



import pyfpgrowth

def fp_find_association_rules(dataset, minsup, minconf):
    patterns = pyfpgrowth.find_frequent_patterns(dataset, minsup*len(dataset))
    rules = pyfpgrowth.generate_association_rules(patterns, minconf)
    return (patterns, rules)

def fp_show_mining_results(ap, N):
    (patterns, rules) = ap
    #print("Frequent Itemsets:\n------------------")
    #for key, val in patterns.items():
    #    print('{}  support = {}'.format(key, round(val/N, 3)))
    #print()
    print("Rules:\n------")
    for key, val in rules.items():
        head = key
        tail = val[0]
        confidence = val[1]
        if len(tail) == 0:
            continue
        print('({}) ==> ({})  confidence = {}'.format(', '.join(head), ', '.join(tail), round(confidence, 3)))
    print()


dat = df_flow[df_flow['station id'] == 519][['in_flow_count', 'out_flow_count']]
dat.head(5)


pd.isnull(dat).sum()


print("Min: {}\nMax: {}".format(dat.values.min(), dat.values.max()))

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15,5))

ax = plt.subplot(2, 2, 1)
dat['in_flow_count'].hist(bins=25)
ax.set_title("in flow count")

ax = plt.subplot(2, 2, 2)
dat['out_flow_count'].hist(bins=25)
ax.set_title("out flow count")

ax = plt.subplot(2, 2, 3)
ax.set_yscale('log')
dat['in_flow_count'].hist(bins=25)
ax.set_title("in flow count (log scale)")

ax = plt.subplot(2, 2, 4)
ax.set_yscale('log')
dat['out_flow_count'].hist(bins=25)
ax.set_title("out flow count (log scale)")

fig.tight_layout()

dat_1 = copy.deepcopy(dat)
dat_1['in_flow_count'] = pd.cut(dat_1['in_flow_count'], bins = 5, \
                                labels = ["in.level-1", "in.level-2", "in.level-3", \
                                          "in.level-4", "in.level-5"]).astype(str)
pd.cut(dat['in_flow_count'], bins = 5).value_counts()

dat_1['out_flow_count'] = pd.cut(dat_1['out_flow_count'], bins = 5, \
                                labels = ["out.level-1", "out.level-2", "out.level-3", \
                                          "out.level-4", "out.level-5"]).astype(str)
pd.cut(dat['out_flow_count'], bins = 5).value_counts()

dat_2 = copy.deepcopy(dat)
dat_2['in_flow_count'] = pd.qcut(dat_2['in_flow_count'], q = 5, \
                                labels = ["in.zero", "in.extreme-low", "in.low", \
                                          "in.medium", "in.high"]).astype(str)
pd.qcut(dat['in_flow_count'], q = 5).value_counts()

dat_2['out_flow_count'] = pd.qcut(dat_2['out_flow_count'], q = 5, \
                                labels = ["out.zero", "out.extreme-low", "out.low", \
                                          "out.medium", "out.high"]).astype(str)
pd.qcut(dat['out_flow_count'], q = 5).value_counts()

print("Apriori\n********")
ap = apriori_find_association_rules(dat_1.values.tolist(), 0.1, 0.2)
apriori_show_mining_results(ap)

print("FP-Growth\n*********")
fp = fp_find_association_rules(dat_1.values.tolist(), 0.1, 0.2)
fp_show_mining_results(fp, dat_1.shape[0])

print("Apriori\n********")
ap = apriori_find_association_rules(dat_2.values.tolist(), 0.1, 0.2)
apriori_show_mining_results(ap)

print("FP-Growth\n*********")
fp = fp_find_association_rules(dat_2.values.tolist(), 0.1, 0.2)
fp_show_mining_results(fp, dat_2.shape[0])


dat = df_flow[df_flow['station id'] == 519][['time', 'in_flow_count', 'out_flow_count']]
dat['flow_count'] = dat['in_flow_count'] + dat['out_flow_count']
dat['time'] = ["{:02d}:{:02d}".format(dt.hour, dt.minute) for dt in dat['time']]
dat = dat[['time', 'flow_count']]
dat.head(5)

dat_1 = copy.deepcopy(dat)
dat_1['time'] = ["{:02d}:00~{:02d}:00".format(math.floor(dt.hour/2)*2, math.floor(dt.hour/2)*2+2) \
                 for dt in pd.to_datetime(dat_1['time'])]
dat_1['time'] = dat_1['time'].astype(str)

dat_1['flow_count'] = pd.qcut(dat_1['flow_count'], q = 3, \
                                labels = ["low", "medium", "high"]).astype(str)

pd.qcut(dat['flow_count'], q = 3).value_counts()



dat_2 = copy.deepcopy(dat)
mapping = ["Night"]*6 + ["Morning"]*5 + ["Noon"]*2 + ["Afternoon"]*3 + ["Evening"]*6 + ["Night"]*2
dat_2['time'] = [mapping[math.floor(dt.hour)] for dt in pd.to_datetime(dat_2['time'])]

#Note that I use the equi-depth approach again to deal
# with the in/out flow counts. (Intervals for low, medium, and high flow counts are shown as belows.)

dat_2['flow_count'] = pd.qcut(dat_2['flow_count'], q = 3, \
                                labels = ["low", "medium", "high"]).astype(str)

pd.qcut(dat['flow_count'], q = 3).value_counts()

print("Apriori\n********")
ap = apriori_find_association_rules(dat_1.values.tolist(), 0.05, 0.6)
apriori_show_mining_results(ap)

print("FP-Growth\n*********")
fp = fp_find_association_rules(dat_1.values.tolist(), 0.05, 0.6)
fp_show_mining_results(fp, dat_1.shape[0])


print("Apriori\n********")
ap = apriori_find_association_rules(dat_2.values.tolist(), 0.05, 0.6)
apriori_show_mining_results(ap)

print("FP-Growth\n*********")
fp = fp_find_association_rules(dat_2.values.tolist(), 0.05, 0.6)
fp_show_mining_results(fp, dat_2.shape[0])
x=0