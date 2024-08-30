#!/usr/bin/env python
# coding: utf-8

# # Future Stock Price Prediction 

# In[ ]:


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


# In[ ]:


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


# In[ ]:


# Step 3: Preparing Future Data for Prediction

# Copy the last row of the current data as the starting point for future predictions
last_row = df_final.iloc[-1][features]

# Repeat the last row for the number of future days to predict
future_data = pd.DataFrame([last_row] * forecast_days, index=future_dates, columns=features)

# Scale the future data
future_data_scaled = scaler.transform(future_data)

print("\nFuture data prepared and scaled for prediction.")


# In[ ]:


# Step 4: Predicting Future Stock Prices

# Predict future prices using both models
future_df['Predicted_Close_LR'] = model.predict(future_data_scaled)  # Using Linear Regression
future_df['Predicted_Close_RF'] = rf_model.predict(future_data_scaled)  # Using Random Forest

# Display the first few future predictions
print("\nFuture stock price predictions (first few days):")
print(future_df.head())


# In[ ]:


print(future_df[['Predicted_Close_LR', 'Predicted_Close_RF']])


# In[ ]:




