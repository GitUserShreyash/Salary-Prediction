

import pandas as pd
df=pd.read_csv('/Salary Data.csv')

df.info()

print(df)

df.isnull().sum()

for col in ['Age', 'Years of Experience', 'Salary']:
    df[col].fillna(df[col].mean(), inplace=True)

for col in ['Gender', 'Education Level', 'Job Title']:
    df[col].fillna(df[col].mode()[0], inplace=True)

display(df.isnull().sum())

from sklearn.preprocessing import LabelEncoder

# Initialize LabelEncoder
le = LabelEncoder()

# Apply Label Encoding to categorical columns
for col in ['Gender', 'Education Level', 'Job Title']:
    df[col] = le.fit_transform(df[col])
    print(f"Column '{col}' encoded. Unique values after encoding: {df[col].unique()}")

display(df.head())

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# Define features (X) and target (y)
X = df.drop('Salary', axis=1)
y = df['Salary']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Data split into training and testing sets.")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")

"""### 1. Linear Regression"""

# Initialize and train the Linear Regression model
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)

# Make predictions
y_pred_lr = linear_reg.predict(X_test)

# Evaluate the model
r2_lr = r2_score(y_test, y_pred_lr)
mae_lr = mean_absolute_error(y_test, y_pred_lr)

print(f"Linear Regression R-squared: {r2_lr:.4f}")
print(f"Linear Regression Mean Absolute Error: {mae_lr:.2f}")

"""### 2. Decision Tree Regressor"""

# Initialize and train the Decision Tree Regressor model
decision_tree_reg = DecisionTreeRegressor(random_state=42)
decision_tree_reg.fit(X_train, y_train)

# Make predictions
y_pred_dt = decision_tree_reg.predict(X_test)

# Evaluate the model
r2_dt = r2_score(y_test, y_pred_dt)
mae_dt = mean_absolute_error(y_test, y_pred_dt)

print(f"Decision Tree Regressor R-squared: {r2_dt:.4f}")
print(f"Decision Tree Regressor Mean Absolute Error: {mae_dt:.2f}")

"""### 3. Random Forest Regressor"""

# Initialize and train the Random Forest Regressor model
random_forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)
random_forest_reg.fit(X_train, y_train)

# Make predictions
y_pred_rf = random_forest_reg.predict(X_test)

# Evaluate the model
r2_rf = r2_score(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)

print(f"Random Forest Regressor R-squared: {r2_rf:.4f}")
print(f"Random Forest Regressor Mean Absolute Error: {mae_rf:.2f}")

"""### 4. Support Vector Regressor (SVM)"""

# Initialize and train the Support Vector Regressor model
# SVM can be sensitive to scaling, so it's often beneficial to scale the data first.
# For simplicity, we'll run it directly, but consider scaling for better performance.
svr_reg = SVR(kernel='rbf') # Using RBF kernel as a common starting point
svr_reg.fit(X_train, y_train)

# Make predictions
y_pred_svr = svr_reg.predict(X_test)

# Evaluate the model
r2_svr = r2_score(y_test, y_pred_svr)
mae_svr = mean_absolute_error(y_test, y_pred_svr)

print(f"Support Vector Regressor R-squared: {r2_svr:.4f}")
print(f"Support Vector Regressor Mean Absolute Error: {mae_svr:.2f}")

"""### 5. K-Nearest Neighbors Regressor (KNN)"""

# Initialize and train the K-Nearest Neighbors Regressor model
knn_reg = KNeighborsRegressor(n_neighbors=5) # Using 5 neighbors as a common starting point
knn_reg.fit(X_train, y_train)

# Make predictions
y_pred_knn = knn_reg.predict(X_test)

# Evaluate the model
r2_knn = r2_score(y_test, y_pred_knn)
mae_knn = mean_absolute_error(y_test, y_pred_knn)

print(f"K-Nearest Neighbors Regressor R-squared: {r2_knn:.4f}")
print(f"K-Nearest Neighbors Regressor Mean Absolute Error: {mae_knn:.2f}")



"""### Model Performance Comparison"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Create a DataFrame to store model performance metrics
model_performance = pd.DataFrame({
    'Model': ['Linear Regression', 'Decision Tree', 'Random Forest', 'SVM', 'KNN'],
    'R-squared': [r2_lr, r2_dt, r2_rf, r2_svr, r2_knn],
    'MAE': [mae_lr, mae_dt, mae_rf, mae_svr, mae_knn]
})

display(model_performance.sort_values(by='R-squared', ascending=False))

# Plot R-squared scores
plt.figure(figsize=(12, 6))
sns.barplot(x='Model', y='R-squared', data=model_performance.sort_values(by='R-squared', ascending=False), palette='viridis', hue='Model', legend=False)
plt.title('Model Comparison: R-squared Scores')
plt.ylabel('R-squared Score')
plt.ylim(min(0, model_performance['R-squared'].min() - 0.1), 1) # Ensure y-axis starts from 0 or slightly below min R-squared
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Plot Mean Absolute Error (MAE) scores
plt.figure(figsize=(12, 6))
sns.barplot(x='Model', y='MAE', data=model_performance.sort_values(by='MAE', ascending=True), palette='magma', hue='Model', legend=False)
plt.title('Model Comparison: Mean Absolute Error (MAE)')
plt.ylabel('Mean Absolute Error')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd

# Define features (X) and target (y)
X = df.drop('Salary', axis=1)
y = df['Salary']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train models
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'SVM': SVR(kernel='rbf'),
    'KNN': KNeighborsRegressor(n_neighbors=5)
}

# Store performance metrics
performance_metrics = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    performance_metrics.append({
        'Model': name,
        'R-squared': r2,
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse
    })

# Create a DataFrame for performance comparison
model_comparison_df = pd.DataFrame(performance_metrics)

display(model_comparison_df.sort_values(by='R-squared', ascending=False))

"""### Visualizing Comprehensive Model Performance"""

import matplotlib.pyplot as plt
import seaborn as sns

# Plot R-squared scores
plt.figure(figsize=(12, 6))
sns.barplot(x='Model', y='R-squared', data=model_comparison_df.sort_values(by='R-squared', ascending=False), palette='viridis', hue='Model', legend=False)
plt.title('Model Comparison: R-squared Scores')
plt.ylabel('R-squared Score')
plt.ylim(min(0, model_comparison_df['R-squared'].min() - 0.1), 1) # Ensure y-axis starts from 0 or slightly below min R-squared
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Plot Mean Absolute Error (MAE) scores
plt.figure(figsize=(12, 6))
sns.barplot(x='Model', y='MAE', data=model_comparison_df.sort_values(by='MAE', ascending=True), palette='magma', hue='Model', legend=False)
plt.title('Model Comparison: Mean Absolute Error (MAE)')
plt.ylabel('Mean Absolute Error')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Plot Mean Squared Error (MSE) scores
plt.figure(figsize=(12, 6))
sns.barplot(x='Model', y='MSE', data=model_comparison_df.sort_values(by='MSE', ascending=True), palette='coolwarm', hue='Model', legend=False)
plt.title('Model Comparison: Mean Squared Error (MSE)')
plt.ylabel('Mean Squared Error')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Plot Root Mean Squared Error (RMSE) scores
plt.figure(figsize=(12, 6))
sns.barplot(x='Model', y='RMSE', data=model_comparison_df.sort_values(by='RMSE', ascending=True), palette='cubehelix', hue='Model', legend=False)
plt.title('Model Comparison: Root Mean Squared Error (RMSE)')
plt.ylabel('Root Mean Squared Error')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

"""### Combined Model Performance Visualization"""

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(18, 12))

# Subplot 1: R-squared scores
plt.subplot(2, 2, 1) # 2 rows, 2 columns, 1st plot
sns.barplot(x='Model', y='R-squared', data=model_comparison_df.sort_values(by='R-squared', ascending=False), palette='viridis', hue='Model', legend=False)
plt.title('Model Comparison: R-squared Scores')
plt.ylabel('R-squared Score')
plt.ylim(min(0, model_comparison_df['R-squared'].min() - 0.1), 1)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Subplot 2: Mean Absolute Error (MAE) scores
plt.subplot(2, 2, 2) # 2 rows, 2 columns, 2nd plot
sns.barplot(x='Model', y='MAE', data=model_comparison_df.sort_values(by='MAE', ascending=True), palette='magma', hue='Model', legend=False)
plt.title('Model Comparison: Mean Absolute Error (MAE)')
plt.ylabel('Mean Absolute Error')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Subplot 3: Mean Squared Error (MSE) scores
plt.subplot(2, 2, 3) # 2 rows, 2 columns, 3rd plot
sns.barplot(x='Model', y='MSE', data=model_comparison_df.sort_values(by='MSE', ascending=True), palette='coolwarm', hue='Model', legend=False)
plt.title('Model Comparison: Mean Squared Error (MSE)')
plt.ylabel('Mean Squared Error')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Subplot 4: Root Mean Squared Error (RMSE) scores
plt.subplot(2, 2, 4) # 2 rows, 2 columns, 4th plot
sns.barplot(x='Model', y='RMSE', data=model_comparison_df.sort_values(by='RMSE', ascending=True), palette='cubehelix', hue='Model', legend=False)
plt.title('Model Comparison: Root Mean Squared Error (RMSE)')
plt.ylabel('Root Mean Squared Error')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

"""These comprehensive plots reinforce that the Random Forest Regressor consistently outperforms other models across all metrics (highest R-squared, lowest MAE, MSE, and RMSE). The SVM model's significantly higher error metrics across the board further highlights its poor fit without proper scaling or hyperparameter tuning.

These plots clearly show that the Random Forest Regressor has the highest R-squared and lowest MAE, indicating it's the best-performing model among those tested for this dataset. The SVM model performs significantly worse, highlighting the need for data scaling before applying SVM.
"""
