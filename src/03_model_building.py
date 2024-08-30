#!/usr/bin/env python
# coding: utf-8

# # Model Building and Training

# In[ ]:


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


# In[ ]:


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


# In[ ]:


# Step 3: Training a Simple Model (Linear Regression)

from sklearn.linear_model import LinearRegression

# Initialize the Linear Regression model
model = LinearRegression()

# Train the model on the scaled training data
model.fit(X_train_scaled, y_train)

# Display the coefficients of the model
print("\nModel coefficients:")
print(dict(zip(features, model.coef_)))


# In[ ]:


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


# In[ ]:


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


# In[ ]:


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

