# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
url = "https://raw.githubusercontent.com/datasets/house-prices-uk/master/data/data.csv"
data = pd.read_csv(url)

# Display the first few rows of the dataset
print("Dataset preview:")
print(data.head())

# Data preprocessing
# Remove rows with missing values
data.dropna(inplace=True)

# Select features and target variable
X = data[['bedrooms', 'bathrooms', 'sq_ft', 'year_built', 'location']]
y = data['price']

# Convert categorical variables into dummy/indicator variables
X = pd.get_dummies(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Evaluate the model
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

print("\nTrain RMSE:", train_rmse)
print("Test RMSE:", test_rmse)

# Plot actual vs. predicted prices for test data
plt.scatter(y_test, y_pred_test)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs. Predicted House Prices (Test Data)")
plt.show()
