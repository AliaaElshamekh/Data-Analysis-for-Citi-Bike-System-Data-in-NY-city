import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import *
import seaborn as sns
from matplotlib import rcParams
import datetime as dt
from geopy.distance import vincenty

df = pd.read_csv("201910-citibike-tripdata.csv")
df.head()
df.info()
print (df.shape)
df.isnull().sum()
df = df.dropna(axis=1, how='any')
df.shape


#Percentage of missing data.
def missing_data(df):
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum()/df.isnull().count()*100).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return missing_data
missing_data(df)



#Ensure data is formatted correctly to avoid errors in the visuals
df['starttime'] = to_datetime(df['starttime'])
df['stoptime'] = to_datetime(df['stoptime'])
df['start station name'] = df['start station name'].astype('category')
df['end station name'] = df['end station name'].astype('category')
df['usertype'] = df['usertype'].astype('category')
df['gender'] = df['gender'].astype('category')
round(df.describe(),2)

#Quasi Confirm Hypothesis in point #3
df_bikenum = pd.DataFrame()
df_bikenum['First Bike'] = df[df['tripduration'] < 90]['start station name']
df_bikenum['Second Bike'] = df[df['tripduration'] < 90]['end station name']
#df_bikenum.head()


#Clear up enviornment and drop double count
del(df_bikenum)
df = df.drop(df.index[(df['tripduration'] < 90) &
                          (df['start station latitude'] == df['end station latitude'])])



#Data for Top 5 Stations visual
top5 = pd.DataFrame()
top5['Station']=df['start station name'].value_counts().head().index
top5['Number of Starts']=df['start station name'].value_counts().head().values
top5['Station'] = top5['Station'].cat.remove_unused_categories()
top5['Station'] = top5['Station'].astype('object')


#Plot for Part 1: Top 5 Stations
ax = sns.barplot('Station', 'Number of Starts', data = top5, palette="GnBu_d")
ax.set_title('Top 5 Citi Bike Stations by Number of Starts', fontsize = 12)
rcParams['figure.figsize'] = 12,7
ax.set_xticklabels(ax.get_xticklabels(),rotation=40, ha = 'right')
for index, row in top5.iterrows():
    ax.text(index,row['Number of Starts']-4000,row['Number of Starts'],
            color='white', ha="center", fontsize = 10)
plt.show()


del(top5)
#Drop NA Usertype
df = df.dropna(subset=['usertype'])

#Calculate trip duration
TD_user = pd.DataFrame()
TD_user['Avg. Trip Duration'] = round(df.groupby('usertype')['tripduration'].mean(),2)
TD_user = TD_user.reset_index()
TD_user['usertype'] = TD_user['usertype'].astype('object')

#Average trip Duration per User Type with Anomalies
ax2 = sns.barplot('usertype', 'Avg. Trip Duration', data = TD_user,palette="GnBu_d")
ax2.set_title('Average Trip Duration by User Type (with anomalies)')
#rcParams['figure.figsize'] = 12,7
ax2.set_xticklabels(ax2.get_xticklabels(),rotation=40, ha = 'right')
ax2.set_ylabel('Avg. Trip Duration (Seconds)')
for index, row in TD_user.iterrows():
    ax2.text(index,row['Avg. Trip Duration']-70,(str(row['Avg. Trip Duration'])+"  Seconds"),
             color='white', ha="center", fontsize = 10)
plt.show()

#Boxplots are more informative to visualize breakdown of data
del(TD_user)
df.boxplot('tripduration', by = 'usertype')
plt.show()

#Remove anomalies based on definition above
df = df.drop(df.index[(df['tripduration'] > 7200)])

#Boxplots are more informative to visualize breakdown of data
df.boxplot('tripduration', by = 'usertype')
plt.show()
#Boxplot without outliers
df.boxplot('tripduration', by = 'usertype',showfliers=False)
plt.show()

#Add Minutes column for Trip Duration
df['Minutes'] = df['tripduration']/60
#For Visual purposes, rounded
df['Minutes'] = round(df['Minutes'])
df['Minutes'] = df['Minutes'].astype(int)

#Final Boxplot with some outliers. Could turn of outliers with showfliers = False
df.boxplot('Minutes', by = 'usertype')
plt.show()
df.boxplot('Minutes', by = 'usertype', showfliers = False)
plt.show()

TD_user2 = pd.DataFrame()
TD_user2['Avg. Trip Duration'] = round(df.groupby('usertype')['Minutes'].mean(),1)
TD_user2 = TD_user2.reset_index()
TD_user2['usertype'] = TD_user2['usertype'].astype('object')

#Average Trip Duration Based on Minutes
ax3 = sns.barplot('usertype', 'Avg. Trip Duration', data = TD_user2,palette="GnBu_d")
ax3.set_title('Average Trip Duration by User Type')
#rcParams['figure.figsize'] = 12,10
ax3.set_xticklabels(ax3.get_xticklabels(),rotation=40, ha = 'right')
for index, row in TD_user2.iterrows():
    ax3.text(row.name,row['Avg. Trip Duration']-1,(str(row['Avg. Trip Duration'])+"  Minutes"),
             color='white', ha="center", fontsize = 10)
plt.show()


del(TD_user2)
#Undo rounding for modelling purposes
df['Minutes'] = df['tripduration']/60


#Identify the 10 most popular trips
trips_df = pd.DataFrame()
trips_df = df.groupby(['start station name','end station name']).size().reset_index(name = 'Number of Trips')
trips_df = trips_df.sort_values('Number of Trips', ascending = False)
trips_df["start station name"] = trips_df["start station name"].astype(str)
trips_df["end station name"] = trips_df["end station name"].astype(str)
trips_df["Trip"] = trips_df["start station name"] + " to " + trips_df["end station name"]
trips_df = trips_df[:10]
trips_df = trips_df.drop(['start station name', "end station name"], axis = 1)
trips_df = trips_df.reset_index()
#trips_df.head()

ax4 = sns.barplot('Number of Trips','Trip', data = trips_df,palette="GnBu_d")
ax4.set_title('Most Popular Trips', fontsize = 20)
ax4.set_ylabel("Trip",fontsize=16)
ax4.set_xlabel("Number of Trips",fontsize=16)
for index, row in trips_df.iterrows():
    ax4.text(row['Number of Trips']-220,index,row['Number of Trips'],
             color='white', ha="center",fontsize = 10)
plt.show()


del(trips_df)
#Drop the tail end of birth years 2 standard deviations below the mean
#df['Birth Year'].mean()-(2*df['Birth Year'].std())
df = df.drop(df.index[(df['birth year'] < 1956)])

#In the future, for a dataset of this size, I would consider using the Haversine formula to calculate distance
#if it's faster.
dist = []
for i in range(len(df)):
    dist.append(vincenty(df.iloc[i]['Start Coordinates'],df.iloc[i]['End Coordinates']).miles)

    if (i%1000000==0):
        print(i)


#Reset Index to avoid issues in future calculations
df = df.reset_index()
df = df.drop('index',axis =1)

df['Distance'] = dist

#Calculate age and drop circular/roundtrips
df['Age'] = 2018 - df['birth year']
df['Age'] = df['Age'].astype(int)

df = df.drop(df.index[(df['Distance'] == 0)])

#1
#df[df['Trip Duration']<90]
#2. Followed the same reasoning as behind Birth Year. People in similar locations tend to also work in a similar industry or location
df['Distance'] = df.groupby(['gender','start station id'])['Distance'].transform(lambda x: x.fillna(x.median()))

df['min_mile'] = round(df['Minutes']/df['Distance'], 2)
df['mile_hour'] = round(df['Distance']/(df['Minutes']/60),2)

round(df.describe(),2)
df = df[df['Distance'] < 30]

#3.1-Done in two steps to ensure data integrity, could've used an or statement as well.
df = df[df['mile_hour']<20]
#3.2
df = df[df['mile_hour']> (df['mile_hour'].mean()-(2*df['mile_hour'].std()))]

df1 = df.drop(df.index[(df['gender'] == 0)])
#Rider performance by age and Gender in Min/Mile
fig, ax5 = plt.subplots(figsize=(11,5))
df1.groupby(['Age','gender']).median()['min_mile'].unstack().plot(ax=ax5, color ="bg")
ax5.legend(['Female','Male'])
plt.ylabel('Median Speed (min/mile)')
plt.title('Rider Performance Based on Gender and Age (Median Speed in min/mile)')
plt.show()
#Rider performance by age and Gender in Miles/hr
del([fig,ax5])
fig1, ax6 = plt.subplots(figsize=(11,5))
df1.groupby(['Age','gender']).median()['mile_hour'].unstack().plot(ax=ax6,color ="bg")
ax6.legend(['Female', 'Male'])
plt.ylabel('Median Speed (miles/hr)')
plt.title('Rider Performance Based on Gender and Age (Median Speed in miles/hr)')
plt.show()

#Rider performance by age and Gender in Averge Distance
del([fig1,ax6])
fig2, ax7 = plt.subplots(figsize=(11,5))
df1.groupby(['Age','gender']).mean()['Distance'].unstack().plot(ax=ax7,color ="bg")
ax7.legend(['Female', 'Male'])
plt.ylabel('Average Distance (miles)')
plt.title('Rider Performance Based on Gender and Age (Average Distance in Miles)')
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


#Visual of most used bike based on Number of Trips
ax8 = sns.barplot('Number of Times Used', 'bikeid',data = bike_use_df, palette="GnBu_d")
ax8.set_title('Most Popular Bikes by Number of Times Used')
for index, row in bike_use_df.iterrows():
    ax8.text(row['Number of Times Used']-90,index,row['Number of Times Used'], color='white', ha="center", fontsize =10)
plt.show()

#Bike usage based on minutes used
#del(ax8)
bike_min_df = pd.DataFrame()
bike_min_df['Minutes Used'] = df.groupby('bikeid')['Minutes'].sum()
bike_min_df = bike_min_df.reset_index()
bike_min_df = bike_min_df.sort_values('Minutes Used', ascending = False)
bike_min_df['bikeid'] = bike_min_df['bikeid'].astype(str)
bike_min_df['bikeid'] = ('Bike ' + bike_min_df['bikeid'])
bike_min_df = bike_min_df[:10]
bike_min_df = bike_min_df.reset_index()
#bike_min_df.head()

#Visual of most used bike based on number of minutes used
ax9 = sns.barplot('Minutes Used', 'bikeid',data = bike_min_df, palette="GnBu_d")
ax9.set_title('Most Popular Bikes by Minutes Used')
rcParams['figure.figsize'] = 11,6
for index, row in bike_min_df.iterrows():
    ax9.text(row['Minutes Used']-2800,index,str(round(row['Minutes Used'],2))+' Minutes',
             color='white', ha="center")
plt.show()

x=0