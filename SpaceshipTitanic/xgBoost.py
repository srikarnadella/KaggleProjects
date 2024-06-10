import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Load the training data
train_data = pd.read_csv("train.csv")

# Preprocess the data (fill missing values, convert categorical variables, and scale numerical features)
X = train_data.drop(["PassengerId", "Transported", "Name"], axis=1)
y = train_data["Transported"]

# Fill missing values
imputer = SimpleImputer(strategy='median')
X[['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']] = imputer.fit_transform(X[['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']])

# Convert categorical variables to one-hot encoding
X = pd.get_dummies(X, columns=["HomePlanet", "Cabin", "Destination", "CryoSleep", "VIP"], dummy_na=True)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Hyperparameter tuning for XGBoost
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}
xgb_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Best model from grid search
best_xgb_model = grid_search.best_estimator_

# Make predictions on the validation set
xgb_predictions = best_xgb_model.predict(X_val)

# Evaluate the model
xgb_accuracy = accuracy_score(y_val, xgb_predictions)
print("XGBoost Accuracy:", xgb_accuracy)

# Load the test data
test_data = pd.read_csv("test.csv")
test_passenger_ids = test_data["PassengerId"]

# Preprocess the test data
X_test = test_data.drop(["PassengerId", "Name"], axis=1)
X_test[['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']] = imputer.transform(X_test[['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']])
X_test = pd.get_dummies(X_test, columns=["HomePlanet", "Cabin", "Destination", "CryoSleep", "VIP"], dummy_na=True)
X_test = X_test.reindex(columns=X.columns, fill_value=0)
X_test = scaler.transform(X_test)

# Make predictions on the test data
test_predictions = best_xgb_model.predict(X_test)

test_predictions = test_predictions.astype(bool)


# Output the results to a CSV file
output = pd.DataFrame({"PassengerId": test_passenger_ids, "Transported": test_predictions})
output.to_csv("xgboost_predictionsUpdated.csv", index=False)
