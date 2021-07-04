#!/usr/bin/env python
# coding: utf-8

# # Project Goal : To help optimize marketing expenses of Yandex.Afisha

# ## Download the data and prepare it for analysis 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import datetime


# In[2]:


# Import visit logs data, converting category type and datetime types
visit_logs = pd.read_csv('/datasets/visits_log_us.csv', dtype={'Device':'category'},parse_dates=['Start Ts','End Ts'])
print(visit_logs.info())
print(visit_logs.describe())
visit_logs


# Imported visit logs data, converting category type and datetime types looked general information about the dataset.

# In[3]:


#Finding missing values 
visit_logs.isnull().sum()


# In[4]:


#Finding Duplicates
visit_logs.duplicated().sum()


# In[5]:


# Modify visit_logs column names
visit_logs = visit_logs.rename(columns={"Device": "device_type", "End Ts":"session_end", "Start Ts":"session_start", "Source Id":"source_id","Uid":"user_id"})
visit_logs


# In[6]:


# Import order logs data, converting datetime types
order_logs = pd.read_csv('/datasets/orders_log_us.csv',parse_dates=['Buy Ts'])
print(order_logs.info())
print(order_logs.describe())
order_logs


# Imported order logs data, converting datetime types looked general information about the dataset.

# In[7]:


#Finding missing values 
order_logs.isnull().sum()


# In[8]:


#Finding Duplicates
order_logs.duplicated().sum()


# In[9]:


# Modify order_logs column names
order_logs = order_logs.rename(columns={"Buy Ts":"order_datetime", "Revenue":"revenue","Uid":"user_id"})
order_logs


# In[10]:


# Import costs data, converting datetime types
cost = pd.read_csv('/datasets/costs_us.csv',parse_dates=['dt'])
print(cost.info())
print(cost.describe())
cost


# Imported cost data, converting datetime types looked general information about the dataset.

# In[11]:


#Finding missing values 
cost.isnull().sum()


# In[12]:


#Finding Duplicates
cost.duplicated().sum()


# In[13]:


# Modify cost column names
cost = cost.rename(columns={"dt":"ad_datetime","costs":"ad_expenses"})
cost


# # Make reports and calculate metrics 

# ## Product 

# ###  How many people use it every day, week and month?

# In[14]:


# How many people use it every day, week and month?

# Create product usage from visit logs table
product_usage = visit_logs.copy()

# Separate columsn for day, week, month values
product_usage['session_date'] = product_usage['session_start'].dt.date
product_usage['session_week'] = product_usage['session_start'].dt.week
product_usage['session_month'] = product_usage['session_start'].dt.month
product_usage['session_year'] = product_usage['session_start'].dt.year

# Calculate DAU: number of daily active unique users
dau_total = product_usage.groupby('session_date').agg({"user_id":"nunique"}).reset_index()
dau_total.columns = ['session_date', 'n_users']
dau_avg = dau_total['n_users'].mean()
print(round(dau_avg, 2))

# Calculate WAU: number of weekly active unique users
wau_total = product_usage.groupby('session_week').agg({"user_id":"nunique"}).reset_index()
wau_total.columns = ['session_week', 'n_users']
wau_avg = wau_total['n_users'].mean()
print(round(wau_avg, 2))

# Calculate DAU: number of monthly active unique users
mau_total = product_usage.groupby('session_month').agg({"user_id":"nunique"}).reset_index()
mau_total.columns = ['session_month', 'n_users']
mau_avg = mau_total['n_users'].mean()
print(round(mau_avg,2))


# Daily Total Usage: 907.99 users
# Weekly Total Usage: 5825.29 users
# Monthly Total Usage: 23228.42 users
# 
# It's possible to explore user activity to find out how many unique visits this product gets to give insight as to what extend the service is interesting. The metrics to describe the numbe of active users can be expressed in terms of daily total usage (DAU), weekly total usage (WAU) and monthly total usage (MAU). These vanity metrics can be calculated by counting the average number of unique users for each day, week and months.
# 
# The number of daily active users of Yandex.Afisha is 908, the number of weekly active users is 5825 and the number of monthly active users is 23228.

# In[15]:


import sys
import warnings
if not sys.warnoptions:
       warnings.simplefilter("ignore")


# In[16]:


# plot daily active users
plt.figure(figsize=(8,6));
plt.plot(dau_total['session_date'],dau_total['n_users'])
plt.title('Daily Number of Active Users')
plt.xlabel('Dates')
plt.ylabel('# of Active Users')
plt.show()

# plot weekly active users
plt.figure(figsize=(8,6));
plt.plot(wau_total['session_week'],wau_total['n_users'])
plt.title('Weekly Number of Active Users')
plt.xlabel('Dates')
plt.ylabel('# of Active Users')
plt.show()

# plot monthly active users
plt.figure(figsize=(8,6));
plt.plot(mau_total['session_month'],mau_total['n_users'])
plt.title('Monthly Number of Active Users')
plt.xlabel('Dates')
plt.ylabel('# of Active Users')
plt.show()


# ###  How many sessions are there per day?

# In[17]:


# How many sessions are there per day? (One user might have more than one session.)

# group the visit data by session date and then count the # of unqiue users and # of unique sessions
sessions_per_day = product_usage.groupby('session_date').agg({'user_id':['count','nunique']}).reset_index()
sessions_per_day.columns = ['session_date','n_sessions','n_users']
sessions_per_day_amount = sessions_per_day['n_sessions'].mean()
print('There are ' + str(round(sessions_per_day_amount, 2)) + " sessions per day")

#  Plot sessions per day
plt.figure(figsize=(8,6))
plt.plot(pd.to_datetime(sessions_per_day['session_date']),sessions_per_day['n_sessions'])
plt.title('Sessions per Day')
plt.xlabel('Dates')
plt.ylabel('number of Sessions')
sessions_per_day


# In[18]:


sessions_per_day['n_user_session_per_day']= sessions_per_day['n_sessions'] / sessions_per_day['n_users']
sessions_per_day_user_amount = sessions_per_day['n_user_session_per_day'].mean()
print('There are ' + str(round(sessions_per_day_user_amount, 2)) + " sessions per day")
plt.figure(figsize=(8,6))
plt.plot(pd.to_datetime(sessions_per_day['session_date']),sessions_per_day['n_user_session_per_day'])
plt.title('Sessions per Day')
plt.xlabel('Dates')
plt.ylabel('number of Sessions per user')
sessions_per_day


# To calculate the average number of unique sessions there on any given day, the visit data can be grouped by the date of the session and count the number of unique users and the number of unique sessions.
# There is an average of 987.36 sessions per day for Yandex.Afisha.

# ###  What is the length of each session?

# In[19]:


# What is the length of each session?

# calculate the duration of a session(session_end - session_start)
product_usage['session_duration_sec'] = (product_usage['session_end'] - product_usage['session_start']).dt.seconds

# calculate the average duration
average_length_per_session = product_usage['session_duration_sec'].median()
print('Average length per session:', round(average_length_per_session, 2))

plt.hist(product_usage['session_duration_sec'],range=[0,2000],bins=50)
plt.title('Session Length')
plt.xlabel('Length (secs)')
plt.ylabel('number of Sessions')
plt.show()


# To calculate the average length of each session, we can examine every row in the visit dataframe and find the difference between the datetime of the end of a session and the datetime of the start of a session.
# The average length of each session is 300 seconds (5 mins).

# ###  How often do users come back?

# In[20]:


# How often do users come back?

# calculate sticky factor metrics
sticky_weekly_avg = (dau_avg / wau_avg) * 100
print(str(round(sticky_weekly_avg, 2)) + "% of users on average come back weekly")

sticky_monthly_avg = (dau_avg / mau_avg) * 100
print(str(round(sticky_monthly_avg, 2)) + "% of users on average come back monthly")

# graph sticky factor metrics 
wau_total['sticky_weekly'] = (dau_avg / wau_total['n_users']) * 100
plt.figure(figsize=(8,6))
plt.plot(wau_total['session_week'],wau_total['sticky_weekly'])
plt.title('% of users that come back weekly')
plt.xlabel('Weeks')
plt.ylabel('% of returning users')
plt.show()

mau_total['sticky_monthly'] = (dau_avg / mau_total['n_users']) * 100
plt.figure(figsize=(8,6))
plt.plot(mau_total['session_month'],mau_total['sticky_monthly'])
plt.title('% of users that come back monthly')
plt.xlabel('Months')
plt.ylabel('% of returning users')
plt.show()


# In[21]:


product_usage['session_start'] = pd.to_datetime(product_usage['session_start'])
first_activity_date = product_usage.groupby(['user_id'])['session_start'].min()
first_activity_date.name = 'first_activity_date'
user_activity = product_usage.join(first_activity_date,on='user_id')
print(user_activity.head())

user_activity['first_activity_month'] = user_activity['session_start'].astype('datetime64[M]').dt.date
user_activity['activity_month'] = (pd.to_datetime(user_activity['session_start'],
                                                unit='d')
                                  - pd.to_timedelta(user_activity['session_start'].dt.month,
                                                    unit='d')).dt.date

user_activity['cohort_lifetime'] = user_activity['activity_month'] - user_activity['first_activity_month']
user_activity['cohort_lifetime'] = user_activity['cohort_lifetime'] / np.timedelta64(1,'W')
user_activity['cohort_lifetime'] = user_activity['cohort_lifetime'].astype(int)
cohorts = user_activity.groupby(['first_activity_month', 'cohort_lifetime']).agg({'user_id': 'nunique'}).reset_index()

initial_users_count = cohorts[cohorts['cohort_lifetime'] == 0][
    ['first_activity_month', 'user_id']
]
#print(initial_users_count) 

initial_users_count = initial_users_count.rename(
    columns={'user_id': 'cohort_users'}
) 
cohorts = cohorts.merge(initial_users_count, on='first_activity_month') 
cohorts['retention'] = cohorts['user_id'] / cohorts['cohort_users'] 

print(cohorts['retention'].mean())

retention_pivot = cohorts.pivot_table(
    index='first_activity_month',
    columns='cohort_lifetime',
    values='retention',
    aggfunc='sum',
)

#retention_5 = retention_pivot[[0,1,2,3,4,5]]

import seaborn as sns
from matplotlib import pyplot as plt

sns.set(style='white')
plt.figure(figsize=(15, 10))
plt.title('Cohorts: User Retention')
sns.heatmap(
    retention_pivot, annot=True, fmt='.3%', linewidths=1, linecolor='gray'
)
plt.show()


# To calculate how often users come back, we can calculate the stick factor metrics in terms of weeks and months. This is found by dividing the number of daily active users by the number of weekly active users (to find the number of times users come back weekly) and dividing the number of daily active users by the number of monthly active users (to find the number of times users come back monehtly).
# 
# Almost 16% of users come back weekly and 4% of users come back monthly.

# ##  Sales 

# ###  When do people start buying?

# In[22]:


# When do people start buying?

# for each user, find date of first order
first_order_dates = order_logs.groupby('user_id').agg({'order_datetime':'min'}).reset_index()
first_order_dates.columns= ['user_id','first_order_date']
first_order_dates['first_order_month'] = first_order_dates['first_order_date'].dt.month

# for each user, find date of first session
first_session_dates = visit_logs.groupby('user_id').agg({'session_start':'min'}).reset_index()
first_session_dates.columns= ['user_id','first_session_date']
first_session_dates['first_session_month'] = first_session_dates['first_session_date'].dt.month

# merge tables on user_id
turnover = pd.merge(first_session_dates, first_order_dates, on='user_id')

# calculate time between first session and first order
turnover['turnover_time_days'] = (turnover['first_order_date'] - turnover['first_session_date']).dt.days
print(turnover)

avg_turnover_time = turnover['turnover_time_days'].mean()
print('People start buying ' + str(round(avg_turnover_time,2)) +" days after first registering")


# In[23]:


bins = [0, 5, 10, 20, 30, 90, 180, 400]
labels = ['0-5d','5-10d','10-20d','20-30d','30-90d','90-180d','>180d']
turnover['turnover_cohort'] = pd.cut(x=turnover['turnover_time_days'], bins=bins, labels=labels)


# To discover when people start buying, we can find the date of the first visit/session and first order for each user. By subtracting the date of the first visit from the date of the first order, we can find how long after the first visit the user made an order and purchased the product. We can find the average turnover time with the mean.
# 
# The average turnover time for users from the first visit to the first order for Yandex.Afisha is about 16 days.

# ###  how many users in each cohort?

# In[24]:


# how many users in each cohort
users_per_cohort = turnover.groupby('turnover_cohort')['user_id'].nunique().reset_index()
users_per_cohort.columns = ['turnover_cohort','n_users']

# calculate % of users in each cohort
total_users = users_per_cohort['n_users'].sum()
users_per_cohort['users %'] = (users_per_cohort['n_users'] / total_users) * 100

display(users_per_cohort)


# Most users start buying between 0 to 5 days after the first session at almost 80% of users but thereafter, the largest turnover cohort falls in the 3-6 month window at 6% of users.

# ###  How many orders do they make during a given period of time?

# In[25]:


# How many orders do they make during a given period of time? 

# extract order month for each order
order_logs['order_month'] = order_logs['order_datetime'].astype('datetime64[M]').dt.month

# group by order month and find # of unique users placing orders each month and find # of orders made each month
unique_orders = order_logs.groupby('order_month').agg({'user_id':'nunique','order_datetime':'count'}).reset_index()
unique_orders.columns = ['order_month','n_unique_users','n_orders']

# create col with average monthly orders per month
unique_orders['average_monthly_orders'] = unique_orders['n_orders'] / unique_orders['n_unique_users']

# calculate average monthly purchases
avg_monthly_purchases = round(unique_orders['average_monthly_orders'].mean(), 2)

print('The average number of orders made in any given month is ' + str(avg_monthly_purchases))


# To find how many orders do users make during a period of time, we can start by defining a period of time as a single month. Now the question becomes how many orders do users make during a months time? We can extract the month of each order from the datetime of the order, group the orders by the months, count the # of unique users ordering for each month, count the number of orders made each month and then find the average number of monthly orders by diving the number of orders with the number of unique users.
# 
# The average number of orders made during a month for Yandex.Afisha is 1.22 orders.

# ###  What is the average purchase size?

# In[26]:


# What is the average purchase size?

avg_purchase_size = round(order_logs['revenue'].mean(), 2)
print("The average purchase size is " + str(avg_purchase_size) + " dollars per purchase")


# To find the average purchase size for each order, we can calculate the average of the revenue made from each order.
# 
# The average purchase size for Yandex.Afisha is about $5 per purchase.

# ### LTV 

# In[27]:


# How much money do they bring? (LTV)
# LTV (lifetime value) is the total amount of money a customer brings to the company on average by making purchases

# # copy dataframes
orders = order_logs.copy()
visit = visit_logs.copy()
visit = visit[['session_start','user_id']]

# extract month and date for each datetime row 
orders['order_month'] = orders['order_datetime'].astype('datetime64[M]')
orders['order_date'] = orders['order_datetime'].dt.date

# Calculate when first order for each customer have happened

visits_and_orders = orders.join(visit
                      .sort_values(by='session_start')
                      .groupby(['user_id'])
                      .agg({'session_start': 'min'}),
                      on='user_id', how='inner')

visits_and_orders['first_session_month'] = visits_and_orders['session_start'].astype('datetime64[M]')

# Create cohorts based on first purchase date and revenue

cohort_sizes = visits_and_orders.groupby('first_session_month').agg({'user_id': 'nunique'}).reset_index()
cohort_sizes.rename(columns={'user_id': 'n_buyers'}, inplace=True)
cohorts = visits_and_orders.groupby(['first_session_month', 'order_month']).agg({'revenue': ['sum', 'count']}).reset_index()

# Calculate cohort age

cohorts['age_month'] = (cohorts['order_month'] - cohorts['first_session_month']) / np.timedelta64(1, 'M')
cohorts['age_month'] = cohorts['age_month'].round().astype('int')
cohorts.columns = ['first_session_month', 'order_month', 'revenue', 'n_orders', 'age_month']

# Merge our cohort tables to the final cohort report
report = pd.merge(cohort_sizes, cohorts, on='first_session_month')
report['ltv'] = report['revenue'] / report['n_buyers']

# Create LTV table
ltv_cohort = report.pivot_table(
    index = 'first_session_month',
    columns = 'age_month',
    values = 'ltv',
    aggfunc = 'sum').cumsum(axis=1)

plt.figure(figsize=(15, 5))
ax = sns.heatmap(ltv_cohort, annot=True, fmt='.2f')
ax.set_yticklabels(ltv_cohort.index.strftime('%m-%Y'))
plt.show()


# LTV (lifetime value) is the total amount of money a customer brings to the company on average by making purchases. To calcuate this metric, I extracted the first order month and all months afterwards and grouped this data per user while adding up the revenue. I found the gross profit for each cohort and then calculated LTV from that. The heatmap indicates that the most revenue is always from the first order month and afterwards, the trend tends to decrease so the customer does not bring the company much money after it's initial purchase.
# 
# For the most part, the lifetime value per customer increases over time.

# ## Marketing 

# ###  How much money was spent? 

# In[28]:


# How much money was spent? 

# Overall
cost_overall = cost['ad_expenses'].sum()
print('The overall amount spent on marketing was $' + str(cost_overall))


# ### How much money was spent? Overall/per source/over time 

# In[29]:


# per source
cost_per_source = cost.groupby('source_id').agg({'ad_expenses':'sum'}).reset_index()
cost_per_source.columns = ['source_id','total_ad_expenses']
print(cost_per_source.sort_values('total_ad_expenses',ascending=False))
plt.figure(figsize=(8,6))
plt.bar(cost_per_source['source_id'],cost_per_source['total_ad_expenses'])
plt.title('Ad Costs Per Source')
plt.xlabel('Source')
plt.ylabel('Amount Spent ($)')
plt.show()


# Total cost per source is shown above in the graph. source id 3 has most of the spending.

# In[30]:


# Over time
cost_over_time = cost.copy()
cost_over_time['ad_month'] = cost['ad_datetime'].astype('datetime64[M]').dt.month
cost_over_time = cost_over_time.groupby('ad_month').agg({'ad_expenses':'sum'}).reset_index()
cost_over_time.columns = ['ad_month','ad_expenses']
print(cost_over_time.sort_values('ad_expenses',ascending=False))
plt.figure(figsize=(8,6))
plt.bar(cost_over_time['ad_month'],cost_over_time['ad_expenses'])
plt.title('Ad Costs Over Time')
plt.xlabel('Months')
plt.ylabel('Amount Spent ($)')
plt.show()


# In the december yandex has payed more money on ads which was around $38315.

# ###  How much did customer acquisition from each of the sources cost?

# In[31]:


# How much did customer acquisition from each of the sources cost?

# CAC
# calculate ad expenses per month
monthly_ad_cost = cost.copy()
monthly_ad_cost['ad_month'] = monthly_ad_cost['ad_datetime'].dt.month
monthly_ad_cost = monthly_ad_cost[['ad_month','ad_expenses']]
monthly_ad_cost = monthly_ad_cost.groupby('ad_month')['ad_expenses'].sum().reset_index()

# incorpordate data on costs
report['order_month'] = report['order_month'].dt.month

report = pd.merge(report, monthly_ad_cost, left_on='order_month', right_on='ad_month')

# calculate cac
report['cac'] = report['ad_expenses'] / report['n_buyers']

# starting at first order month and continuing through order months,what is cac per cohort
result = report.pivot_table(index='order_month',columns='ad_month',values='cac',aggfunc='mean').round()
result.fillna('')

plt.figure(figsize=(13,9))
plt.title('CAC per Cohort')
sns.heatmap(result, annot=True, fmt='.1f', linewidths=1, linecolor='green')
plt.show()


# CAC (customer acquisition cost) is the cost of attracting a customer - how much was spent in marketing efforts to acquire a single customer. To calculate this, we can compare the amount spent on ads for the same month that the customer made a purchase by dividing expenses spent on ads by the number of buyers for each month.

# ###  How worthwhile where the investments?

# In[32]:


# How worthwhile where the investments? (ROI)

# calculate return on marketing investment 
report['romi'] = report['ltv'] / report['cac']

output = report.pivot_table(index='order_month', columns='ad_month',values='romi',aggfunc='mean')
output = output.cumsum(axis=1).round(2)
#print(output)

plt.figure(figsize=(13,9))
plt.title('ROI per Cohort')
sns.heatmap(output, annot=True, fmt='.2f', linewidths=1, linecolor='blue')
plt.show()


# The July cohort paid off in the month (ROMI = 0.34). (we start counting at 0.)
# The June and August cohort also paid off.
# In general, by the 6th or 12th month, every cohort had either paid off or gotten close.

# ## Conclusion: advise marketing experts how much money to invest and where. 

# What sources/platforms would you recommend? 
# Source - Source Id 3 to 5
# Back up your choice: what metrics did you focus on? 
# Most users start buying between 0 to 5 days after the first session at almost 80% of users but thereafter, the largest turnover cohort falls in the 3-6 month window at 6% of users.
# Why?
# The July cohort paid off in the month (ROMI = 0.34). (we start counting at 0.) The June and August cohort also paid off. In general, by the 6th or 12th month, every cohort had either paid off or gotten close.
# What conclusions did you draw after finding the metric values?
# Users who ordered for product during the 0th week were more likely to use the app again. Maybe it's because they faced fewer difficulties afterward. It's worth considering how to encourage users to ask for help during the 0th lifetime week.
# The thing is that the company began to grow faster in June. Expenses increased, and so did the number of new customers. But it takes six months to pay off, so profits haven't caught up to expenses yet. The business seems healthy.
# Company need to work on more ad campaigns and advertisment in the first quarter of the year.
