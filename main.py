import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer

# Load train and test datasets
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# Select relevant features
selected_features = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF']

# Separate features and target variable in train data
X_train = train_data[selected_features]
y_train = train_data['SalePrice']

# Separate features in test data
X_test = test_data[selected_features]

# Impute missing values in the test data
imputer = SimpleImputer(strategy='median')
X_test_imputed = imputer.fit_transform(X_test)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the imputed test set
y_pred = model.predict(X_test_imputed)

# Print model performance metrics
print("Model Performance Metrics:")
print("----------------------------")
print("Mean Squared Error (MSE) on Training Data:", mean_squared_error(y_train, model.predict(X_train)))
print("Root Mean Squared Error (RMSE) on Training Data:", np.sqrt(mean_squared_error(y_train, model.predict(X_train))))

# Visualize the relationship between actual and predicted sale prices
plt.figure(figsize=(8, 6))
plt.scatter(y_train, model.predict(X_train), color='orange')
plt.title('Actual vs. Predicted Sale Prices (Training Data)')
plt.xlabel('Actual Sale Price')
plt.ylabel('Predicted Sale Price')
plt.show()
