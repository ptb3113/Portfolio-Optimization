#!/usr/bin/env python
# coding: utf-8

# # Data Preparation and Exploration

# In[ ]:


# Step 1: Loading the Dataset

import pandas as pd

# Load the AAPL stock data from the provided CSV file
df = pd.read_csv('AAPL_stock.csv', index_col='Date', parse_dates=True)

# Display the first few rows to verify the data is loaded correctly
print("First few rows of the dataset:")
df.head()


# In[ ]:


# Step 2: Handling Missing Values

# Check for missing values in the dataset
print("\nChecking for missing values:")
print(df.isnull().sum())

# Drop rows with any missing values (if any)
df_cleaned = df.dropna()

# Verify that missing values are handled
print("\nData after handling missing values:")
df_cleaned.isnull().sum()


# In[ ]:


# Step 3: Calculating Basic Statistics

# Calculate basic statistics like mean, median, standard deviation, etc.
print("\nBasic statistics of the dataset:")
df_cleaned.describe()


# In[ ]:


# Step 4: Visualizing the Stock Price Over Time

import matplotlib.pyplot as plt

# Plot the 'Close' price over time
plt.figure(figsize=(12, 6))
plt.plot(df_cleaned.index, df_cleaned['Close'], label='Close Price', color='blue')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('AAPL Stock Close Price Over Time')
plt.legend()
plt.show()


# In[ ]:


# Step 5: Calculating Moving Averages

# Calculate the 50-day moving average
df_cleaned['MA50'] = df_cleaned['Close'].rolling(window=50).mean()

# Calculate the 200-day moving average
df_cleaned['MA200'] = df_cleaned['Close'].rolling(window=200).mean()

# Display the first few rows to verify the moving averages
print("\nData with moving averages:")
df_cleaned[['Close', 'MA50', 'MA200']].head()


# In[ ]:


# Step 6: Visualizing Moving Averages

# Plot the 'Close' price along with the 50-day and 200-day moving averages
plt.figure(figsize=(12, 6))
plt.plot(df_cleaned.index, df_cleaned['Close'], label='Close Price', color='blue')
plt.plot(df_cleaned.index, df_cleaned['MA50'], label='50-Day MA', color='orange')
plt.plot(df_cleaned.index, df_cleaned['MA200'], label='200-Day MA', color='green')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('AAPL Stock Close Price with Moving Averages')
plt.legend()
plt.show()


# In[ ]:




