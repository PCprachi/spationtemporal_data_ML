# -*- coding: utf-8 -*-
"""EDA.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1g4wf72Q0Z4fbLThcIk_x6IVBB6KvqS5R
"""

# Commented out IPython magic to ensure Python compatibility.
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
import os
import warnings
warnings.filterwarnings('ignore')

df_trn=pd.read_csv("/content/drive/MyDrive/spatiotemporal_trn_data.csv")
df_trn.head

def get_date_dict(date_str):
    reg = re.compile(r"^(?P<yr>\d{4})-(?P<mon>\d{2})-(?P<day>\d{2})T(?P<hr>\d{2}):(?P<min>\d{2}):(?P<sec>\d{2})$")
    m = reg.match(date_str)
    if m:
        return m.groupdict()
    else:
        raise ValueError('The date string does not match the expected format.')

# Convert the 'DATE' column to datetime objects
df_trn['DATE'] = pd.to_datetime(df_trn['DATE'])

# Extract the date components directly without a loop
df_trn['YEAR'] = df_trn['DATE'].dt.year
df_trn['MONTH'] = df_trn['DATE'].dt.month
df_trn['DAY'] = df_trn['DATE'].dt.day
df_trn['HOUR'] = df_trn['DATE'].dt.hour
df_trn['MINUTES'] = df_trn['DATE'].dt.minute
df_trn['SECONDS'] = df_trn['DATE'].dt.second

# Add targets to same df
df_trn['TARGETS'] = pd.read_csv('/content/drive/MyDrive/spatiotemporal_trn_targets.csv', names=['index', 'TARGETS'])['TARGETS']

df_trn.head()

df_trn.shape

df_trn.info()

# unique values in each column
print("\nUnique Values:")
for col in df_trn.columns:
    print(col, df_trn[col].nunique())

# Calculate the minimum and maximum values for all columns
min_values = df_trn.min()
max_values = df_trn.max()

# Print the results
print("Minimum values for each column:")
print(min_values)

print("\nMaximum values for each column:")
print(max_values)

df_trn.describe()

# Value count for each value
for i in df_trn.columns:
    print(i,'\n',df_trn[i].value_counts())
    print('-'*90)

# Print unique values in the 'TARGETS' column
print(df_trn['TARGETS'].unique())
# Assuming df_trn is your DataFrame
df_trn['TARGETS'] = pd.to_numeric(df_trn['TARGETS'], errors='coerce')
plt.figure(figsize=(10, 6))
bin_edges = [0, 20, 40, 60, 80, 100]
plt.hist(df_trn['TARGETS'], bins=bin_edges, color='skyblue', edgecolor='black')
plt.title('Histogram of TARGETS Values')
plt.xlabel('TARGETS')
plt.ylabel('Number of Instances')
plt.xticks(bin_edges)  # Set x-axis ticks to the specified bin edges
plt.grid(True)
plt.show()

# Assuming df_trn is your DataFrame
nan_counts = df_trn.isnull().sum()

# Display the count of NaN values for each column
print("NaN counts for each column:")
print(nan_counts)

df_trn.info()

non_null_counts = df_trn.notnull().sum()

# Display the count of non-null values for each column
print("Non-null counts for each column:")
print(non_null_counts)

