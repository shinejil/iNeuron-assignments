#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

def vandermonde_matrix(input_vector, increasing=False):
    n = len(input_vector)
    if increasing:
        matrix = np.column_stack([input_vector**(n-i-1) for i in range(n)])
    else:
        matrix = np.column_stack([input_vector**i for i in range(n-1, -1, -1)])
    return matrix


# In[2]:


import numpy as np

def moving_average(input_sequence, window_size):
    sequence_length = len(input_sequence)
    num_averages = sequence_length - window_size + 1
    moving_averages = []
    
    for i in range(num_averages):
        average = np.mean(input_sequence[i:i+window_size])
        moving_averages.append(average)
    
    return moving_averages


# In[3]:


input_sequence = [3, 5, 7, 2, 8, 10, 11, 65, 72, 81, 99, 100, 150]
window_size = 3

averages = moving_average(input_sequence, window_size)
print(averages)


# In[ ]:




