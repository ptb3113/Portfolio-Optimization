#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Portfolio Optimization
# By: Pratham Brahmbhatt


# Portfolio Optimization with Predictive Analytics
# 
# Project Scope and Objectives
# - Objective: Forecast AAPL stock prices using historical data and predictive models.
# - Focus: Single-stock analysis (AAPL) with an aim to optimize investment decisions.
# - Outcome: A comprehensive analysis with predictive models and visualizations.
# 

# # Data Preparation and Exploration

# In[1]:


# Step 1: Loading the Dataset

import pandas as pd

# Load the AAPL stock data from the provided CSV file
df = pd.read_csv('AAPL_stock.csv', index_col='Date', parse_dates=True)

# Display the first few rows to verify the data is loaded correctly
print("First few rows of the dataset:")
df.head()


# In[2]:


# Step 2: Handling Missing Values

# Check for missing values in the dataset
print("\nChecking for missing values:")
print(df.isnull().sum())

# Drop rows with any missing values (if any)
df_cleaned = df.dropna()

# Verify that missing values are handled
print("\nData after handling missing values:")
df_cleaned.isnull().sum()


# In[3]:


# Step 3: Calculating Basic Statistics

# Calculate basic statistics like mean, median, standard deviation, etc.
print("\nBasic statistics of the dataset:")
df_cleaned.describe()


# In[4]:


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


# In[5]:


# Step 5: Calculating Moving Averages

# Calculate the 50-day moving average
df_cleaned['MA50'] = df_cleaned['Close'].rolling(window=50).mean()

# Calculate the 200-day moving average
df_cleaned['MA200'] = df_cleaned['Close'].rolling(window=200).mean()

# Display the first few rows to verify the moving averages
print("\nData with moving averages:")
df_cleaned[['Close', 'MA50', 'MA200']].head()


# In[6]:


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





# # Feature Engineering

# In[7]:


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


# In[8]:


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


# In[9]:


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





# # Model Building and Training

# In[18]:


# Step 1: Splitting the Data into Training and Testing Sets

from sklearn.model_selection import train_test_split

# Define the features (X) and the target variable (y)
# We'll use all the engineered features except the target variable 'Close'
features = ['MA50', 'MA200', 'Daily_Return', 'Volatility', 'RSI', 'MACD', 'Signal_Line']
X = df_final[features]
y = df_final['Close']

# Split the data into 80% training and 20% testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Display the shapes of the resulting datasets
print(f"Training set shape: X_train={X_train.shape}, y_train={y_train.shape}")
print(f"Testing set shape: X_test={X_test.shape}, y_test={y_test.shape}")


# In[19]:


# Step 2: Scaling the Features

from sklearn.preprocessing import StandardScaler

# Initialize the scaler
scaler = StandardScaler()

# Fit the scaler on the training data and transform both training and testing data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Display the first few rows of the scaled training data to verify
print("\nFirst few rows of the scaled training data:")
pd.DataFrame(X_train_scaled, columns=features).head()


# In[20]:


# Step 3: Training a Simple Model (Linear Regression)

from sklearn.linear_model import LinearRegression

# Initialize the Linear Regression model
model = LinearRegression()

# Train the model on the scaled training data
model.fit(X_train_scaled, y_train)

# Display the coefficients of the model
print("\nModel coefficients:")
print(dict(zip(features, model.coef_)))


# In[21]:


# Step 4: Evaluating the Model

from sklearn.metrics import mean_squared_error, r2_score

# Predict on the test set
y_pred = model.predict(X_test_scaled)

# Calculate Mean Squared Error (MSE) and R-squared (R2)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display the evaluation metrics
print(f"\nMean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R2): {r2:.2f}")


# In[22]:


# Step 5: Visualizing the Model's Predictions

import matplotlib.pyplot as plt

# Plot the actual vs predicted prices
plt.figure(figsize=(12, 6))
plt.plot(y_test.index, y_test, label='Actual Prices', color='blue')
plt.plot(y_test.index, y_pred, label='Predicted Prices', linestyle='--', color='orange')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Actual vs Predicted AAPL Stock Prices (Linear Regression)')
plt.legend()
plt.show()


# In[23]:


# Step 6: Experimenting with a More Complex Model (Random Forest)

from sklearn.ensemble import RandomForestRegressor

# Initialize the Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model on the scaled training data
rf_model.fit(X_train_scaled, y_train)

# Predict on the test set using Random Forest
y_pred_rf = rf_model.predict(X_test_scaled)

# Calculate and display the evaluation metrics for Random Forest
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f"\nRandom Forest - Mean Squared Error (MSE): {mse_rf:.2f}")
print(f"Random Forest - R-squared (R2): {r2_rf:.2f}")


# In[ ]:





# # Future Stock Price Prediction 

# In[32]:


# Step 1: Retraining the Model on the Entire Dataset

# Use the entire dataset to retrain the model for future predictions
X_full = df_final[features]
y_full = df_final['Close']

# Scale the entire dataset using the previously fitted scaler
X_full_scaled = scaler.fit_transform(X_full)

# Retrain the Linear Regression model on the entire dataset
model.fit(X_full_scaled, y_full)

# Retrain the Random Forest model on the entire dataset
rf_model.fit(X_full_scaled, y_full)

print("Models retrained on the entire dataset for future predictions.")


# In[33]:


# Step 2: Generating Future Dates for Prediction

# Define the number of future days you want to predict
forecast_days = 30

# Generate future dates (this assumes daily frequency)
# Replace 'closed' with 'inclusive'
last_date = df_final.index[-1]
future_dates = pd.date_range(last_date, periods=forecast_days + 1, inclusive='right')

# Create a DataFrame to store future predictions
future_df = pd.DataFrame(index=future_dates)

print(f"\nFuture dates generated from {last_date} to {future_dates[-1]}.")


# In[34]:


# Step 3: Preparing Future Data for Prediction

# Copy the last row of the current data as the starting point for future predictions
last_row = df_final.iloc[-1][features]

# Repeat the last row for the number of future days to predict
future_data = pd.DataFrame([last_row] * forecast_days, index=future_dates, columns=features)

# Scale the future data
future_data_scaled = scaler.transform(future_data)

print("\nFuture data prepared and scaled for prediction.")


# In[35]:


# Step 4: Predicting Future Stock Prices

# Predict future prices using both models
future_df['Predicted_Close_LR'] = model.predict(future_data_scaled)  # Using Linear Regression
future_df['Predicted_Close_RF'] = rf_model.predict(future_data_scaled)  # Using Random Forest

# Display the first few future predictions
print("\nFuture stock price predictions (first few days):")
print(future_df.head())


# In[40]:


print(future_df[['Predicted_Close_LR', 'Predicted_Close_RF']])


# In[ ]:





# # Risk Management

# Step 1: Calculate Key Risk Metrics

# In[41]:


# Step 1.1: Calculating Value at Risk (VaR)

import numpy as np

# Calculate daily returns
df_final['Daily_Return'] = df_final['Close'].pct_change()

# Calculate the Value at Risk (VaR) at a 95% confidence level
VaR_95 = np.percentile(df_final['Daily_Return'].dropna(), 5)

# Display the VaR
print(f"Value at Risk (VaR) 95%: {VaR_95:.4f}")


# In[42]:


# Step 1.2: Calculating the Sharpe Ratio

# Assume an annual risk-free rate of 1%
risk_free_rate = 0.01  

# Calculate the Sharpe Ratio
sharpe_ratio = (df_final['Daily_Return'].mean() - risk_free_rate / 252) / df_final['Daily_Return'].std()

# Display the Sharpe Ratio
print(f"Sharpe Ratio: {sharpe_ratio:.4f}")


# Step 2: Scenario Analysis and Stress Testing

# In[43]:


# Step 2.1: Performing Scenario Analysis

# Example scenario: Simulate a market crash
# Assume a market crash as a drop of 3 standard deviations below the mean
market_crash_return = df_final['Daily_Return'].mean() - 3 * df_final['Daily_Return'].std()

# Display the simulated market crash return
print(f"Simulated Market Crash Return: {market_crash_return:.4f}")


# In[44]:


# Step 2.2: Performing Stress Testing

# Example stress test: Calculate the portfolio value after a market crash
# Assuming an initial portfolio value (e.g., $1,000,000)
initial_portfolio_value = 1000000

# Calculate the portfolio value after a simulated market crash
portfolio_value_after_crash = initial_portfolio_value * (1 + market_crash_return)

# Display the portfolio value after the stress test
print(f"Portfolio Value After Market Crash: ${portfolio_value_after_crash:,.2f}")


# In[ ]:





# In[ ]:




