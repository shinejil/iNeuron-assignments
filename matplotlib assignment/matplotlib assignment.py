#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# In[15]:



# Define the periodic function
def periodic_func(x, a, b, c, d):
    return a + b * np.cos(2 * np.pi * x / 12 - c) + d * np.sin(2 * np.pi * x / 12 - c)

# Given data
months = np.arange(1, 13)
max_temp = np.array([39, 41, 43, 47, 49, 51, 45, 38, 37, 29, 27, 25])
min_temp = np.array([21, 23, 27, 28, 32, 35, 31, 28, 21, 19, 17, 18])

# Fit the data to the periodic function
popt_max, pcov_max = curve_fit(periodic_func, months, max_temp)
popt_min, pcov_min = curve_fit(periodic_func, months, min_temp)

# Generate x-values for plotting the fit
x = np.linspace(1, 12, 100)

# Plot the data and the fit
plt.figure(figsize=(8, 6))
plt.scatter(months, max_temp, label='Max Temperature', color='red')
plt.scatter(months, min_temp, label='Min Temperature', color='blue')
plt.plot(x, periodic_func(x, *popt_max), color='orange', label='Max Temp Fit')
plt.plot(x, periodic_func(x, *popt_min), color='cyan', label='Min Temp Fit')
plt.xlabel('Month')
plt.ylabel('Temperature (°C)')
plt.title('Temperature Variation in a City (India)')
plt.legend()
plt.show()


# <b>titanic data ploting</b>

# In[5]:



url="https://raw.githubusercontent.com/Geoyi/Cleaning-Titanic-Data/master/titanic_original.csv"
data = pd.read_csv(url)


# In[6]:


data


# In[7]:


gender_counts = data['sex'].value_counts()
gender_counts


# In[9]:


labels = gender_counts.index
sizes = gender_counts.values


# In[10]:


plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, autopct='%1.1f%%')
plt.title('Male/Female Proportion')
plt.axis('equal')
plt.show()


# In[11]:


male_data = data[data['sex'] == 'male']
female_data = data[data['sex'] == 'female']


# In[12]:


plt.figure(figsize=(10, 6))
plt.scatter(male_data['fare'], male_data['age'], c='blue', label='Male', alpha=0.5)
plt.scatter(female_data['fare'], female_data['age'], c='red', label='Female', alpha=0.5)
plt.xlabel('Fare')
plt.ylabel('Age')
plt.title('Fare paid vs Age')
plt.legend()
plt.show()


# In[ ]:




