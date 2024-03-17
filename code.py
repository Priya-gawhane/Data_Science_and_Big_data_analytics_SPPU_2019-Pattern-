#!/usr/bin/env python
# coding: utf-8

# NAME: Priya Gawhane
# 
# ASSIGNMENT-1
# 
# Perform the following operations using Python on any open source dataset (e.g., data.csv)
# 
# 1.Import all the required Python Libraries.
# Locate an open source data from the web (https://www.kaggle.com/competitions/fake-news/data?select=train.csv). Provide a. clear description of the data and its source (i.e., URL of the web site).
# 
# 2.Load the Dataset into pandas data frame.
# 
# 3.Data Preprocessing: check for missing values in the data using pandas insull(), describe() function to get some initial statistics. Provide variable descriptions. Types of variables etc. Check the dimensions of the data frame.
# 
# 4.Data Formatting and Data Normalization: Summarize the types of variables by checking the data types (i.e., character, numeric, integer, factor, and logical) of the variables in the data set. If variables are not in the correct data type, apply proper type conversions.
# 
# 5.Turn categorical variables into quantitative variables in Python. In addition to the codes and outputs, explain every operation that you do in the above steps and explain everything that you do to import/read/scrape the data set.

# install the dependecies(libraries)
# 
# !pip install pandas
# 
# !pip install numpy
# 
# !pip install matplotlib.pyplot

# # Data Wrangling

# data wrangling is the process of transforming raw data into a more usable format for analytics or machine learning.

# In[1]:


#importing the dependecies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# #data collection and preprocessing
# df = pd.read_csv('train.csv')

# In[2]:


df = pd.read_csv('autodata.csv')


# In[3]:


#getting the first 5 rows and columns of the dataset
df.head(5)


# In[4]:


#getting the last 5 rows and columns af the dataset
df.tail(5)


# In[5]:


#getting the sample of dataset
df.sample(8)


# In[6]:


#checking the info of the given dataset
df.info()


# In[7]:


df.describe()


# In[8]:


df.isnull()


# Evaluating for Missing Data
# 
# The missing values are converted to Python's default. We use Python's built-in functions to identify these missing values. There are two methods to detect missing data:
# 
# isnull()
# .notnull()
# The output is a boolean value indicating whether the value that is passed into the argument is in fact missing data. "True" stands for missing value, while "False" stands for not missing value.
# 
# Deal with missing data
# 
# Drop data
# Drop the whole row
# Drop the whole column
# Replace data
# Replace it by mean
# Replace it by frequency / mode
# Replace it based on other functions

# In[9]:


df.isnull().sum()


# In[10]:


df.isnull()


# In[11]:


df.notnull()


# In[13]:


df.notnull().sum()


# Based on the summary above, each column has 205 rows of data, seven columns containing missing data:
# 
# stroke : 4 missing data
# 
# horsepower: 2 missing data
# 
# peak-rpm: 2 missing data
# 
# horsepower-binned: 2 missing data

# In[14]:


# calculate the mean vaule for "stroke" column
avg_stroke = df["stroke"].astype("float").mean(axis = 0)
print("Average of stroke:", avg_stroke)

# replace NaN by mean value in "stroke" column
df["stroke"].replace(np.nan, avg_stroke, inplace = True)


# In[19]:


#Calculate the mean value for the 'horsepower' column:
avg_hp = df["horsepower"].astype("float").mean(axis = 0)
print("Average of stroke:", avg_hp)


# In[20]:


#Replace "NaN" by mean value:
df["horsepower"].replace(np.nan, avg_hp, inplace = True)


# In[21]:


#Calculate the mean value for 'peak-rpm' column:
avg_rpm = df["peak-rpm"].astype("float").mean(axis = 0)
print("Average of stroke:", avg_rpm)


# In[22]:


#Replace NaN by mean value:
df["peak-rpm"].replace(np.nan, avg_hp, inplace = True)
df['num-of-doors'].value_counts()


# In[23]:


df['num-of-doors'].value_counts().idxmax()


# In[24]:


#replace the missing 'num-of-doors' values by the most frequent 
df["num-of-doors"].replace(np.nan, "four", inplace=True)

# simply drop whole row with NaN in "horsepower-binned" column
df.dropna(subset=["horsepower-binned"], axis=0, inplace=True)

# reset index, because we droped two rows
df.reset_index(drop=True, inplace=True)


# In[26]:


#checking for the missing values again
df.isnull().sum()


# Data Standardization
# Data is usually collected from different agencies with different formats. (Data Standardization is also a term for a particular type of data normalization, where we subtract the mean and divide by the standard deviation)
# 
# What is Standardization?
# 
# Standardization is the process of transforming data into a common format which allows the researcher to make the meaningful comparison.
let us perform standardization on the following dataset

Transform mpg to L/100km:

In our dataset, the fuel consumption columns "city-mpg" and "highway-mpg" are represented by mpg (miles per gallon) unit. Assume we are developing an application in a country that accept the fuel consumption with L/100km standard

# In[27]:


df['city-L/100km'] = 235/df["city-mpg"]
df.head()


# In[28]:


df['highway-L/100km'] = 235/df["highway-mpg"]
df.head()
     


# # Data Normalization
# 
# what is Normalization?
# 
# Normalization refers to rescaling real-valued numeric attributes into a 0 to 1 range. Data normalization is used in machine learning to make model training less sensitive to the scale of features

# let us perform Normalization on the following dataset
# 
# To demonstrate normalization, let's say we want to scale the columns "length", "width" and "height"
# 
# Target:would like to Normalize those variables so their value ranges from 0 to 1.
# 

# In[30]:


df['length'] = df['length']/df['length'].max()
df['width'] = df['width']/df['width'].max()


# In[31]:


df['height'] = df['height']/df['height'].max() 
df[["length","width","height"]].head()


# # Indicator variable (or dummy variable)
# 
# An indicator variable (or dummy variable) is a numerical variable used to label categories. They are called 'dummies' because the numbers themselves don't have inherent meaning.
# 
# Why we use indicator variables?
# 
# So we can use categorical variables for regression analysis in the later modules.
# 
# Example:
# 
# We see the column "fuel-type" has two unique values, "gas" or "diesel". Regression doesn't understand words, only numbers. To use this attribute in regression analysis, we convert "fuel-type" into indicator variables.
# 
# We will use the panda's method 'get_dummies' to assign numerical values to different categories of fuel type.

# In[32]:


df.columns


# In[33]:


df['aspiration'].value_counts()


# In[34]:


dummy_variable_1 = pd.get_dummies(df["aspiration"])
dummy_variable_1.head()


# In[35]:


df = pd.concat([df, dummy_variable_1], axis=1)
df.drop("aspiration", axis = 1, inplace=True)


# In[36]:


df.head()


# # Binning
# Binning is a process of transforming continuous numerical variables into discrete categorical 'bins', for grouped analysis.
# 
# Example:
# 
# In our dataset, "horsepower" is a real valued variable ranging from 48 to 288, it has 57 unique values. What if we only care about the price difference between cars with high horsepower, medium horsepower, and little horsepower (3 types)? Can we rearrange them into three ‘bins' to simplify analysis?

# # horsepower

# In[38]:


df["horsepower"]=df["horsepower"].astype(float, copy=True)


# In[39]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as plt
from matplotlib import pyplot
plt.pyplot.hist(df["horsepower"])

plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")


# In[40]:


bins = np.linspace(min(df["horsepower"]), max(df["horsepower"]), 4)
bins


# In[41]:


group_names = ['Low', 'Medium', 'High']


# In[42]:


df['horsepower-binned'] = pd.cut(df['horsepower'], bins, labels=group_names, include_lowest=True )
df[['horsepower','horsepower-binned']].head(20)


# In[43]:


df["horsepower-binned"].value_counts()


# In[46]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as plt
from matplotlib import pyplot
pyplot.bar(group_names, df["horsepower-binned"].value_counts())

# set x/y labels and plot title
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")


# # Peak-RPM

# In[47]:


df["peak-rpm"]=df["peak-rpm"].astype(float, copy=True)


# In[48]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as plt
from matplotlib import pyplot
plt.pyplot.hist(df["peak-rpm"])

plt.pyplot.xlabel("peak-rpm")
plt.pyplot.ylabel("count")
plt.pyplot.title("Peak-rpm bins")


# In[50]:


bins = np.linspace(min(df["peak-rpm"]), max(df["peak-rpm"]), 4)
bins


# In[51]:


group_names1 = ['Low', 'Medium', 'High']


# In[52]:


df['peakrpm-binned'] = pd.cut(df['peak-rpm'], bins, labels=group_names, include_lowest=True )
df[['peak-rpm','peakrpm-binned']].head(20)


# In[53]:


df["peakrpm-binned"].value_counts()


# In[54]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as plt
from matplotlib import pyplot
pyplot.bar(group_names, df["peakrpm-binned"].value_counts())

# set x/y labels and plot title
plt.pyplot.xlabel("Peak-rpm")
plt.pyplot.ylabel("count")
plt.pyplot.title("peak-rpm bins")


# # Wheel-base

# In[55]:


df["wheel-base"]=df["wheel-base"].astype(float, copy=True)


# In[56]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as plt
from matplotlib import pyplot
plt.pyplot.hist(df["wheel-base"])

plt.pyplot.xlabel("wheel-base")
plt.pyplot.ylabel("count")
plt.pyplot.title("Wheel-base bins")


# In[57]:


bins = np.linspace(min(df["wheel-base"]), max(df["wheel-base"]), 4)
bins


# In[58]:


group_names = ['Low', 'Medium', 'High']


# In[59]:


df['wheelbase-binned'] = pd.cut(df['wheel-base'], bins, labels=group_names, include_lowest=True )
df[['wheel-base','wheelbase-binned']].head(20)


# In[60]:


df["wheelbase-binned"].value_counts()


# In[61]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as plt
from matplotlib import pyplot
pyplot.bar(group_names, df["wheelbase-binned"].value_counts())

# set x/y labels and plot title
plt.pyplot.xlabel("Wheelbase")
plt.pyplot.ylabel("count")
plt.pyplot.title("Wheelbase bins")


# In[ ]:


S

