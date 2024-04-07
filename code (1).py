#!/usr/bin/env python
# coding: utf-8
Name: Priya Gawhane

Data Wrangling 

II Create an “Academic performance” dataset of students and perform the following operations using Python.

Scan all variables for missing values and inconsistencies. If there are missing values and/or inconsistencies, use any of the suitable techniques to deal with them.

Scan all numeric variables for outliers. If there are outliers, use any of the suitable techniques to deal with them.

Apply data transformations on at least one of the variables. The purpose of this transformation should be one of the following reasons: to change the scale for better understanding of the variable, to convert a non-linear relation into a linear one, or to decrease the skewness and convert the distribution into a normal distribution. Reason and document your approach properly.
# In[1]:


#importing the dependecies
import pandas as pd
import numpy as np 
import seaborn as sns


# In[2]:


df = pd.read_csv('StudentsPerformance.csv')


# In[3]:


df


# In[4]:


#checking for the missing values
df.isnull().sum()

hence there is no missing values
# In[5]:


df.boxplot()


# In[16]:


idf = df[df["math score"] > 25]

