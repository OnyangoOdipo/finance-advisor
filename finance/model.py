import pandas as pd
import seaborn as sb
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv(r'finance\data\kenyan_professionals_expenditures.csv')

# Calculate total expenditure
expenditure_columns = ['housing', 'food', 'transport', 'bills', 'clothing', 'personal_needs', 'debt_repayment', 'family_needs', 'health_insurance', 'entertainment_leisure']
df['total_expenditure'] = df[expenditure_columns].sum(axis=1)

# Check for missing values
print(df.isnull().sum())

# Create dummy variables for categorical columns
df = pd.get_dummies(df, columns=['gender', 'profession'], drop_first=True)

# Initialize the scaler
scaler = StandardScaler()

# Split the data into features and target
X = df.drop(['id', 'total_expenditure'], axis=1)
y = df['total_expenditure']

# Scale the features
X_scaled = scaler.fit_transform(X)

# Save the scaler
with open(r'./finance/models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Save the column names used during training
with open(r'./finance/models/columns.pkl', 'wb') as f:
    pickle.dump(X.columns.tolist(), f)

# Convert scaled features back to DataFrame
X = pd.DataFrame(X_scaled, columns=X.columns)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = RandomForestRegressor(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Random Forest Regressor:")
print(f"  MAE: {mae}")
print(f"  MSE: {mse}")
print(f"  R-squared: {r2}")

# Plotting
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs. Predicted Values')
plt.show()

residuals = y_test - y_pred

plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals, color='green')
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.axhline(y=0, color='red', linestyle='--')
plt.show()

importances = model.feature_importances_
features = X.columns

plt.figure(figsize=(10, 8))
indices = np.argsort(importances)[::-1]
plt.bar(range(X.shape[1]), importances[indices], align='center')
plt.xticks(range(X.shape[1]), features[indices], rotation=90)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importances')
plt.show()

from sklearn.tree import plot_tree

tree_to_plot = model.estimators_[0]

plt.figure(figsize=(20, 10))
plot_tree(tree_to_plot, feature_names=X.columns.tolist(), filled=True, rounded=True, fontsize=10)
plt.title("Decision Tree from Random Forest")
plt.show()

# Save the model to a file in pkl format
with open(r'./finance/models/random_model.pkl', 'wb') as f:
    pickle.dump(model, f)
