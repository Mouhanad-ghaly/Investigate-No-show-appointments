#!/usr/bin/env python
# coding: utf-8

# > **Tip**: Welcome to the Investigate a Dataset project! You will find tips in quoted sections like this to help organize your approach to your investigation. Once you complete this project, remove these **Tip** sections from your report before submission. First things first, you might want to double-click this Markdown cell and change the title so that it reflects your dataset and investigation.
# 
# # Project: Investigate No-show appointments
# 
# ## Table of Contents
# <ul>
# <li><a href="#intro">Introduction</a></li>
# <li><a href="#wrangling">Data Wrangling</a></li>
# <li><a href="#eda">Exploratory Data Analysis</a></li>
# <li><a href="#conclusions">Conclusions</a></li>
# </ul>

# <a id='intro'></a>
# ## Introduction
# 
# ### Dataset Description 
# 
#   In the healthcare field numerous data is produced including patients and disease and the main target is to benefit the patient. In this data set - No show appointments- which collects information from 100k medical appointments in Brazil and is focused on the question of whether or not patients show up for their appointment. A number of characteristics about the patient are included in each row.
# ● ‘ScheduledDay’ tells us on
# what day the patient set up their
# appointment.
# ● ‘Neighborhood’ indicates the
# location of the hospital.
# ● ‘Scholarship’ indicates
# whether or not the patient is
# enrolled in Brasilian welfare
# program Bolsa Família.
# ● Be careful about the encoding
# of the last column: it says ‘No’ if
# the patient showed up to their
# appointment, and ‘Yes’ if they
# did not show up
# 
# 
# ### Question(s) for Analysis
# - What is distribution of age group among non showed people and which mean age is greater male or females
# - Which patient group doesn't attend the most
# - Does recieved sms affect showing in appointment
# - Which day of week has most appointments and which has the most of non showing
# 

# In[1]:


# Use this cell to set up import statements for all of the packages that you
#   plan to use.

# Remember to include a 'magic word' so that your visualizations are plotted
#   inline with the notebook. See this page for more:
#   http://ipython.readthedocs.io/en/stable/interactive/magics.html
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Upgrade pandas to use dataframe.explode() function. 
get_ipython().system('pip install --upgrade pandas==0.25.0')


# <a id='wrangling'></a>
# ## Data Wrangling
# 
# > **Tip**: In this section of the report, you will load in the data, check for cleanliness, and then trim and clean your dataset for analysis. Make sure that you **document your data cleaning steps in mark-down cells precisely and justify your cleaning decisions.**
# 
# 
# 

# In[3]:


# Load your data and print out a few lines. Perform operations to inspect data
#   types and look for instances of missing or possibly errant data.
df = pd.read_csv('noshowappointments-kagglev2-may-2016.csv')


# In[4]:


df.head()


# In[5]:


# cheking for information of each column
df.info()


# In[6]:


# Determine number of rows and columns
df.shape


# In[7]:


# identifying types of data columns
df.dtypes


# In[8]:


# cheking for any empty values
df.isnull().sum()


# In[9]:


#checking for duplicated values 
df.duplicated().sum()


# 
# ### Data Cleaning
# > **Tip**: Make sure that you keep your reader informed on the steps that you are taking in your investigation. Follow every code cell, or every set of related code cells, with a markdown cell to describe to the reader what was found in the preceding cell(s). Try to make it so that the reader can then understand what they will be seeing in the following cell(s).
#  

# In[10]:


# rebaning these columns to their correct spelling 
df.rename(columns = {'Hipertension': 'Hypertension','Handcap':'Handicap'},inplace=True)


# In[11]:


# removing patientid and appointmentid columns as we won't use them
df.drop(['PatientId', 'AppointmentID'], axis=1, inplace=True)


# In[12]:


# checking for negative values which i encountered on making graphs so added this part
x = df[ (df['Age'] < 0) ].index
df.drop(x,inplace=True)


# In[13]:


# making column name in lowercase to make it easy to work with
df.columns = df.columns.str.lower()


# In[14]:


# changing type of appointmentday and scheduledday columns to datetime to be easy to work with
df['scheduledday'] = pd.to_datetime(df['scheduledday'],format='%Y-%m-%d %H:%M:%S')
df['appointmentday'] = pd.to_datetime(df['appointmentday'],format='%Y-%m-%d %H:%M:%S')
df['num_days'] = (df['appointmentday']-df['scheduledday']).dt.days
df.num_days.head(10)


# In[15]:


# making a new column for number of days
df.num_days = np.where(df.num_days<0, 0, df.num_days)
df.num_days.head(10)


# In[16]:


df['num_days'].unique()


# In[17]:


#checking that datatypes changed correctly
df.dtypes


# <a id='eda'></a>
# ## Exploratory Data Analysis
# 
#  explorations.
# 
# 
# ### Research Question 1 (What is distribution of age group among non showed people and which mean age is greater male or females)

# In[18]:


df.groupby('no-show')['age'].plot(kind='hist',legend=True,bins=15)


# In this plot the distribution of age against appointment show in which Yes is people who didn't show and No is people who showed

# In[19]:


sns.catplot(x='age',y='no-show',data=df,kind='box',col='gender')


# In this box plot the mean age of who didn't show is younger and mean age of non showing females is older than males 
# Outliers is also shown

# ### Research Question 2  (Which patient group doesn't attend the most?)

# In[20]:


df.groupby('no-show')[['hypertension','diabetes','alcoholism','handicap']].sum().plot(kind='bar')


# This plot show that hypertension is most common patient group and it is the highest for no show for appointments may indicate that hypertension patient doesn't feel ill so attend appointment on time 

# ### Research Question 3  (Does recieved sms affect showing in appointment?)

# In the next plot the patients who didn't show and recieved sms against the count of sms

# In[21]:


df.query('sms_received== 1').groupby('no-show')['sms_received'].count().plot(kind='bar', xlabel='no show and recieved sms')


# In the next plot the patients who didn't show against the count of patients didn't recieve sms

# In[22]:


df.query('sms_received== 0').groupby('no-show')['sms_received'].count().plot(kind='bar', xlabel='no show and didn\'t recieve sms')


# Conclusion in the last two graphs, 
# in spite of patients non showing at appointment in both cases whether recieved or didn't recieve is nearly equal but on comparing the proportion of both, the sms has lowered the proportion of non showing at appointment

# ### Research Question 4  (Which day of week has most appointments and which has the most of non showing ?)

# In[23]:


# making a new column with day names of appointment
df['Day'] = df['appointmentday'].dt.day_name()


# In[24]:


#checking it worked
df.head(2)


# In[25]:


#making a bar plot for days of week against appointment days count
df.Day.value_counts().sort_values().plot(kind='bar',xlabel= 'Days of the week',ylabel='Count of appointment days')


# In the previous plot the most common appointment day shown is wednsday

# In[26]:


# replacing '-' with '_' so the query function work
df.columns =[column.replace("-", "_") for column in df.columns]


# In[27]:


# subsetting the group of people who didn't in appointment
df_yes = df.query('no_show == "Yes"')
df_yes.head()


# In[28]:


#forming a plot of day column which we made previously against count of non showing in each day 
df_yes.groupby('Day').count()['no_show'].sort_values().plot(kind='bar')


# In the previous plot it is concluded that tuesday is the most common day of week to have non showing appointments

# <a id='conclusions'></a>
# ## Conclusions
# 
# > For conclusion of this dataset many columns and factors can help us predict what cause the non showing of patients on appointment day including patients group as hypertension disease, day of the week with most non showing , sms recieving and its decreasing of non showing and the distribution of non showing groups according to age and gender
# 
# > LImitations:
# - The data isn't collected properly the no show columns is misleading with its answers
# - Same patient may be added with same appointment id but different dates 
# 
# > 
# 
# > 
# 
# 

# In[ ]:




