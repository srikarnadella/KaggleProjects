import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

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

# Define a function to create the Neural Network model (used for hyperparameter tuning)
def create_model(optimizer='adam', activation='relu', dropout_rate=0.0):
    model = keras.Sequential([
        layers.Dense(64, activation=activation, input_shape=(X_train.shape[1],)),
        layers.Dropout(dropout_rate),
        layers.Dense(32, activation=activation),
        layers.Dropout(dropout_rate),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Create a KerasClassifier
model = KerasClassifier(build_fn=create_model, verbose=0)

# Hyperparameter tuning for Neural Network
param_grid = {
    'optimizer': ['adam', 'rmsprop'],
    'activation': ['relu', 'tanh'],
    'dropout_rate': [0.0, 0.1, 0.2],
    'batch_size': [32, 64],
    'epochs': [20, 30]
}
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Best model from grid search
best_nn_model = grid_search.best_estimator_

# Make predictions on the validation set
nn_predictions = best_nn_model.predict(X_val)

# Evaluate the model
nn_accuracy = accuracy_score(y_val, nn_predictions)
print("Improved Neural Network Accuracy:", nn_accuracy)

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
test_predictions = best_nn_model.predict(X_test)

# Output the results to a CSV file
output = pd.DataFrame({"PassengerId": test_passenger_ids, "Transported": test_predictions})
output.to_csv("improved_neural_network_predictions.csv", index=False)
