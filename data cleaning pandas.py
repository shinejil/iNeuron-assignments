#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

# Create the initial DataFrame
df = pd.DataFrame({'From_To': ['LoNDon_paris', 'MAdrid_miLAN', 'londON_StockhOlm',
                               'Budapest_PaRis', 'Brussels_londOn'],
                   'FlightNumber': [10045, np.nan, 10065, np.nan, 10085],
                   'RecentDelays': [[23, 47], [], [24, 43, 87], [13], [67, 32]],
                   'Airline': ['KLM(!)', '<Air France> (12)', '(British Airways. )',
                               '12. Air France', '"Swiss Air"']})

# Task 1: Fill in missing FlightNumber values and convert the column to integer
df['FlightNumber'] = df['FlightNumber'].interpolate().astype(int)

# Task 2: Split the From_To column into separate columns
temp_df = pd.DataFrame(df['From_To'].str.split('_', expand=True))
temp_df.columns = ['From', 'To']

# Task 3: Standardize the capitalization of city names in the temporary DataFrame
temp_df['From'] = temp_df['From'].str.capitalize()
temp_df['To'] = temp_df['To'].str.capitalize()

# Task 4: Remove the From_To column from df and attach the temporary DataFrame
df = df.drop('From_To', axis=1)
df = pd.concat([temp_df, df], axis=1)

# Task 5: Expand the RecentDelays column into separate columns
delays = df['RecentDelays'].apply(pd.Series)
delays.columns = ['delay_' + str(i+1) for i in range(delays.shape[1])]
df = df.drop('RecentDelays', axis=1)
df = pd.concat([df, delays], axis=1)

# Print the final DataFrame
print(df)

