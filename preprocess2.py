import pandas as pd
import json
import  datetime

import  re
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import altair as alt
from sklearn import datasets
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
from sklearn.feature_selection import VarianceThreshold
import scipy.stats as sc
import sklearn
from geopy.distance import vincenty
from sklearn import linear_model
import warnings
from sklearn import metrics
warnings.filterwarnings('ignore')
import itertools


df = pd.read_csv("201910-citibike-tripdata.csv")
df.head()
df.info()
print (df.shape)
df.isnull().sum()
df = df.dropna(axis=1, how='any')
df.shape



t1 = df[['start station id', 'start station name', 'start station latitude', 'start station longitude']] \
            .drop_duplicates().rename(columns = {'start station id':'station id', \
                                                 'start station name':'station name', \
                                                 'start station latitude':'station latitude',
                                                 'start station longitude': 'station longitude'})
t2 = df[['end station id', 'end station name', 'end station latitude', 'end station longitude']] \
        .drop_duplicates().rename(columns = {'end station id':'station id', \
                                             'end station name':'station name', \
                                             'end station latitude':'station latitude', \
                                             'end station longitude': 'station longitude'})
df_loc = pd.concat([t1, t2]).drop_duplicates()

df = df[df['start station id']!=3036]
df = df[df['end station id']!=3036]
df_loc = df_loc[df_loc['station id']!=3036]
df_loc.to_csv("station_information.csv", index=None)
df = df[df['tripduration'] <= 24*60*60*20]

# format example: 2017-07-01 00:00:00
df['starttime'] = pd.to_datetime(df['starttime'], format='%Y-%m-%d %H:%M:%S')
df['stoptime'] =pd.to_datetime(df['stoptime'], format='%Y-%m-%d %H:%M:%S')
df.info()
def gen_time_segment(dt):
    if dt.minute < 30:
        minute = "%02d" % 0
    else:
        minute = "%02d" % 30
    return "{}-{}-{} {}:{}".format(dt.year, dt.month, dt.day, dt.hour, minute)

df['start_seg'] = [gen_time_segment(dt) for dt in df['starttime']]
df['stop_seg'] = [gen_time_segment(dt) for dt in df['stoptime']]

df[['start station id', 'starttime', 'start_seg', 'end station id', 'stoptime', 'stop_seg']].head()

inflow = df[['end station id', 'stop_seg']] \
            .groupby(['end station id', 'stop_seg']) \
            .size().reset_index(name='counts') \
            .rename(columns={'end station id':'station id','stop_seg':'time', 'counts':'in_flow_count'})


outflow = df[['start station id', 'start_seg']] \
            .groupby(['start station id', 'start_seg']) \
            .size().reset_index(name='counts') \
            .rename(columns={'start station id':'station id','start_seg':'time', 'counts':'out_flow_count'})
station_id_list = list(df_loc['station id'])

# Create combinations of time series and station ids
time_seg_list = list(pd.date_range("2019-10-01 00:00:00", "2019-10-31 23:30:00", freq="30min"))
template = pd.DataFrame(list(itertools.product(station_id_list, time_seg_list)), \
                        columns=["station id", "time"])

# Merge in/out flow information & Add zeros to missing data according to every time segment
dat = pd.merge(inflow, outflow, on=['station id', 'time'], how='outer')
dat['time'] = pd.to_datetime(dat['time'], format='%Y-%m-%d %H:%M')
dat = dat.merge(template, on=["station id", "time"], how="right").fillna(0)
dat.head()

dat.to_csv("station_flow.csv", index=None)
print("{} stations are found in this dataset.".format(len(station_id_list)))


# Create dictionaries for station latitude/longitude
lat_dic = {}
lon_dic = {}
for index, row in df_loc.iterrows():
    lat_dic[row['station id']] = row['station latitude']
    lon_dic[row['station id']] = row['station longitude']


#Add Minutes column for Trip Duration
df['Minutes'] = df['tripduration']/60
#For Visual purposes, rounded
df['Minutes'] = round(df['Minutes'])
df['Minutes'] = df['Minutes'].astype(int)



# Generate combinations of pairs of station
c = itertools.combinations(station_id_list, 2)
df = df.drop(df.index[(df['birth year'] < 1956)])
# Calculate the averge distance of pairs of stations
dist = 0
dist1=[]
count = 0
for stn1, stn2 in c:
        dist += vincenty((lat_dic[stn1], lon_dic[stn1]), (lat_dic[stn2], lon_dic[stn2])).meters
        count += 1
print("The average distance between different stations is {} (meters)".format(dist/count))

for i in range(len(df)):
    dist1.append(vincenty((df.iloc[i]['start station latitude'], df.iloc[i]['start station longitude']), (df.iloc[i]['end station latitude'], df.iloc[i]['end station longitude'])).meters)

    if (i % 1000000 == 0):
        print(i)



#Reset Index to avoid issues in future calculations
df = df.reset_index()
df = df.drop('index',axis =1)


df['Distance'] = dist1
df['Age'] = 2019 - df['birth year']
df['Age'] = df['Age'].astype(int)

df = df.drop(df.index[(df['Distance'] == 0)])

#1
#df[df['Trip Duration']<90]
#2. Followed the same reasoning as behind Birth Year. People in similar locations tend to also work in a similar industry or location
df['Distance'] = df.groupby(['gender','start station id'])['Distance'].transform(lambda x: x.fillna(x.median()))

df['min_meter'] = round(df['Minutes']/df['Distance'], 2)
df['meter_hour'] = round(df['Distance']/(df['Minutes']/60),2)

round(df.describe(),2)
df = df[df['Distance'] < 50]

#3.1-Done in two steps to ensure data integrity, could've used an or statement as well.
df = df[df['meter_hour']<30]
#3.2
df = df[df['meter_hour']> (df['meter_hour'].mean()-(2*df['meter_hour'].std()))]

df1 = df.drop(df.index[(df['gender'] == 0)])
#Rider performance by age and Gender in Min/Mile
fig, ax5 = plt.subplots(figsize=(11,5))
df1.groupby(['Age','gender']).median()['min_meter'].unstack().plot(ax=ax5, color ="bg")
ax5.legend(['Female','Male'])
plt.ylabel('Median Speed (min/m)')
plt.title('Rider Performance Based on Gender and Age (Median Speed in min/m)')
plt.show()
#Rider performance by age and Gender in Miles/hr
del([fig,ax5])
fig1, ax6 = plt.subplots(figsize=(11,5))
df1.groupby(['Age','gender']).median()['meter_hour'].unstack().plot(ax=ax6,color ="bg")
ax6.legend(['Female', 'Male'])
plt.ylabel('Median Speed (m/hr)')
plt.title('Rider Performance Based on Gender and Age (Median Speed in m/hr)')
plt.show()

#Rider performance by age and Gender in Averge Distance
del([fig1,ax6])
fig2, ax7 = plt.subplots(figsize=(11,5))
df1.groupby(['Age','gender']).mean()['Distance'].unstack().plot(ax=ax7,color ="bg")
ax7.legend(['Female', 'Male'])
plt.ylabel('Average Distance (meters)')
plt.title('Rider Performance Based on Gender and Age (Average Distance in Meters)')
plt.show()

#Bike usage based on number of times used
del(df1)
bike_use_df = pd.DataFrame()
bike_use_df = df.groupby(['bikeid']).size().reset_index(name = 'Number of Times Used')
bike_use_df = bike_use_df.sort_values('Number of Times Used', ascending = False)
#bike_use_df.to_csv('Q5.csv')
bike_use_df = bike_use_df[:10]
bike_use_df['bikeid'] = bike_use_df['bikeid'].astype(str)
bike_use_df['bikeid'] = ('Bike ' + bike_use_df['bikeid'])
bike_use_df = bike_use_df.reset_index()
#bike_use_df.head()



# What are the top stations 3 frequent stations pairs (start station, end station) in weekdays, how about in weekends?
# Split the dataframe into weekdays information & weekends information
df_weekdays = df[df['starttime'].dt.dayofweek < 5]
df_weekends = df[df['starttime'].dt.dayofweek >= 5]

# Count and sort station pair frequencies
stn_pair_weekdays = df_weekdays[['start station id', 'end station id']] \
    .groupby(['start station id', 'end station id']) \
    .size().reset_index(name='counts') \
    .set_index(['start station id', 'end station id']) \
    .sort_values(by='counts', ascending=False)
stn_pair_weekends = df_weekends[['start station id', 'end station id']] \
    .groupby(['start station id', 'end station id']) \
    .size().reset_index(name='counts') \
    .set_index(['start station id', 'end station id']) \
    .sort_values(by='counts', ascending=False)

# Find the top 3 station pairs for weekday & weekend
top_weekday_pair = list(stn_pair_weekdays.head(3).index)
top_weekend_pair = list(stn_pair_weekends.head(3).index)

# Print out the result
print("The top 3 frequent stations pairs in weekdays are: {}, {}, and {}.".format(*top_weekday_pair))
print("The top 3 frequent stations pairs in weekends are: {}, {}, and {}.".format(*top_weekend_pair))

# Sort the average in/out flow count of each station
average_inflow = dat[['station id', 'in_flow_count']] \
    .groupby(['station id']) \
    .mean() \
    .sort_values(by='in_flow_count', ascending=False)
average_outflow = dat[['station id', 'out_flow_count']] \
    .groupby(['station id']) \
    .mean() \
    .sort_values(by='out_flow_count', ascending=False)

# List the top 3 stations
top_inflow = list(average_inflow.head(3).index)
top_outflow = list(average_outflow.head(3).index)

# Print out the result
print("The top 3 stations with highest outflow are: {}, {}, and {}".format(*top_outflow))
print("The top 3 stations with highest inflow are: {}, {}, and {}".format(*top_inflow))

# Popular stations
# Sum up in/out flow at each time station
dat['flow_count'] = dat['in_flow_count'] + dat['out_flow_count']

# Calculate and sort the average flow count for each station
average_flow = dat[['station id', 'flow_count']] \
    .groupby(['station id']) \
    .mean() \
    .sort_values(by='flow_count', ascending=False)

# Find the top 1 station
top_flow = list(average_inflow.head(1).index)

# Print out the result
print("The most popular station is: {}".format(*top_outflow))
# Select station & add information in missing time
small_df = dat[dat['station id'] == 519].sort_values(by='time')
small_df = small_df.sort_values(by='time')

# Plot line chart
small_df.plot(x='time', y=['in_flow_count', 'out_flow_count'], kind='line', figsize=(15,15))
plt.show()

dist = metrics.pairwise_distances([small_df['in_flow_count']], [small_df['out_flow_count']], metric='euclidean')
print("The euclidean distance between in-flow and out-flow of this station is: {}".format(dist[0][0]))


# Calculate variance
small_df['in_flow_variance'] = small_df['in_flow_diff'] / small_df['in_flow_count'].std()
small_df['out_flow_variance'] = small_df['out_flow_diff'] / small_df['out_flow_count'].std()

# Plot line chart
small_df.plot(x='time', y=['in_flow_variance', 'out_flow_variance'], kind='line', figsize=(15,15))
plt.show()


dist = metrics.pairwise_distances([small_df['in_flow_variance']], [small_df['out_flow_variance']], metric='euclidean')
print("The euclidean distance between in-flow and out-flow of this station is: {}".format(dist[0][0]))


# Prepare input for linear regression model
small_df.sort_values(by='time', ascending=True)
length = small_df.shape[0]
time = np.arange(length).reshape(length, 1)

# Create and fit the models
reg_A = linear_model.LinearRegression()
reg_A.fit(time, list(small_df['in_flow_count']))
reg_B = linear_model.LinearRegression()
reg_B.fit(time, list(small_df['out_flow_count']))

# Save the prediction results
small_df['in_flow_linear'] = y_pred_A = reg_A.predict(time)
small_df['out_flow_linear'] = y_pred_B = reg_B.predict(time)

# Plot the fit reuslt
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,10))
ax = plt.subplot(1, 2, 1)
plt.scatter(time, small_df['in_flow_count'].values.reshape(length, 1), color='black', alpha=0.1)
plt.plot(time, y_pred_A, color='blue', linewidth=3)
ax = plt.subplot(1, 2, 2)
plt.scatter(time, small_df['out_flow_count'].values.reshape(length, 1), color='black', alpha=0.1)
plt.plot(time, y_pred_B, color='blue', linewidth=3)


# Calculate distance to the line drawn by linear model
small_df['in_flow_ols'] = small_df['in_flow_count'] - small_df['in_flow_linear']
small_df['out_flow_ols'] = small_df['out_flow_count'] - small_df['out_flow_linear']

# Plot line chart
small_df.plot(x='time', y=['in_flow_ols', 'out_flow_ols'], kind='line', figsize=(15,15))
plt.show()


dist = metrics.pairwise_distances([small_df['in_flow_ols']], [small_df['out_flow_ols']], metric='euclidean')
print("The euclidean distance between in-flow and out-flow of this station is: {}".format(dist[0][0]))

x=0