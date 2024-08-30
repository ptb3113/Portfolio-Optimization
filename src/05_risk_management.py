#!/usr/bin/env python
# coding: utf-8

# # Risk Management

# Step 1: Calculate Key Risk Metrics

# In[ ]:


# Step 1.1: Calculating Value at Risk (VaR)

import numpy as np

# Calculate daily returns
df_final['Daily_Return'] = df_final['Close'].pct_change()

# Calculate the Value at Risk (VaR) at a 95% confidence level
VaR_95 = np.percentile(df_final['Daily_Return'].dropna(), 5)

# Display the VaR
print(f"Value at Risk (VaR) 95%: {VaR_95:.4f}")


# In[ ]:


# Step 1.2: Calculating the Sharpe Ratio

# Assume an annual risk-free rate of 1%
risk_free_rate = 0.01  

# Calculate the Sharpe Ratio
sharpe_ratio = (df_final['Daily_Return'].mean() - risk_free_rate / 252) / df_final['Daily_Return'].std()

# Display the Sharpe Ratio
print(f"Sharpe Ratio: {sharpe_ratio:.4f}")


# Step 2: Scenario Analysis and Stress Testing

# In[ ]:


# Step 2.1: Performing Scenario Analysis

# Example scenario: Simulate a market crash
# Assume a market crash as a drop of 3 standard deviations below the mean
market_crash_return = df_final['Daily_Return'].mean() - 3 * df_final['Daily_Return'].std()

# Display the simulated market crash return
print(f"Simulated Market Crash Return: {market_crash_return:.4f}")


# In[ ]:


# Step 2.2: Performing Stress Testing

# Example stress test: Calculate the portfolio value after a market crash
# Assuming an initial portfolio value (e.g., $1,000,000)
initial_portfolio_value = 1000000

# Calculate the portfolio value after a simulated market crash
portfolio_value_after_crash = initial_portfolio_value * (1 + market_crash_return)

# Display the portfolio value after the stress test
print(f"Portfolio Value After Market Crash: ${portfolio_value_after_crash:,.2f}")


# In[ ]:




