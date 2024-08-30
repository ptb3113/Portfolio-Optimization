#!/usr/bin/env python
# coding: utf-8

# # Feature Engineering

# In[ ]:


# Step 1: Calculating Additional Technical Indicators

# We'll calculate more technical indicators commonly used in stock analysis

# 1.1 Calculate Daily Returns
df_cleaned['Daily_Return'] = df_cleaned['Close'].pct_change()

# 1.2 Calculate Volatility (Rolling Standard Deviation)
df_cleaned['Volatility'] = df_cleaned['Daily_Return'].rolling(window=21).std()  # 21 trading days ~ 1 month

# 1.3 Calculate Relative Strength Index (RSI)
def calculate_RSI(data, window):
    delta = data.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    RS = gain / loss
    RSI = 100 - (100 / (1 + RS))
    return RSI

df_cleaned['RSI'] = calculate_RSI(df_cleaned['Close'], window=14)  # 14-day RSI

# 1.4 Calculate Moving Average Convergence Divergence (MACD)
df_cleaned['MACD'] = df_cleaned['Close'].ewm(span=12, adjust=False).mean() - df_cleaned['Close'].ewm(span=26, adjust=False).mean()

# 1.5 Calculate Signal Line (9-Day EMA of MACD)
df_cleaned['Signal_Line'] = df_cleaned['MACD'].ewm(span=9, adjust=False).mean()

# Display the first few rows to verify the new features
print("Data with additional technical indicators:")
df_cleaned[['Close', 'Daily_Return', 'Volatility', 'RSI', 'MACD', 'Signal_Line']].head()


# In[ ]:


# Step 2: Exploring Feature Importance and Correlation

import seaborn as sns
import matplotlib.pyplot as plt

# 2.1 Calculate the correlation matrix
correlation_matrix = df_cleaned[['Close', 'Daily_Return', 'Volatility', 'RSI', 'MACD', 'Signal_Line', 'MA50', 'MA200']].corr()

# 2.2 Visualize the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix of Technical Indicators')
plt.show()

# 2.3 Display the correlation matrix
print("\nCorrelation Matrix:")
correlation_matrix


# In[ ]:


# Step 3: Dropping or Refining Features

# Based on the correlation analysis, you may decide to drop or refine features.

# For example, if 'Volatility' and 'Daily_Return' are highly correlated, you might choose to keep only one.
# Let's assume we decide to keep all features for now.

# Dropping rows with any missing values (again, since new features may introduce NaNs)
df_final = df_cleaned.dropna()

# Display the final dataset with selected features
print("\nFinal dataset with selected features:")
df_final.head()


# In[ ]:




