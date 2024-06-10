import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from bayes_opt import BayesianOptimization
from datetime import datetime
from sklearn.impute import SimpleImputer
from sklearn.ensemble import StackingRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge

# Load datasets
traindf = pd.read_csv('train.csv')
testdf = pd.read_csv('test.csv')

# Feature Engineering
def preprocess_data(data):
    # Handle missing values
    data.fillna({
        'LotFrontage': data['LotFrontage'].median(),
        'MasVnrArea': 0,
        'GarageYrBlt': data['YearBuilt'],
    }, inplace=True)
    
    for col in data.select_dtypes(include='object').columns:
        data[col].fillna('None', inplace=True)
    
    for col in data.select_dtypes(include='number').columns:
        data[col].fillna(0, inplace=True)

    # Encode categorical features
    le = LabelEncoder()
    for col in data.select_dtypes(include='object').columns:
        data[col] = le.fit_transform(data[col])

    return data

traindf = preprocess_data(traindf)
test_data = preprocess_data(testdf)

# Separate features and target
X = traindf.drop(columns=['SalePrice', 'Id'])
y = traindf['SalePrice']
X_test = test_data.drop(columns=['Id'])

# Split train and validation set
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Define evaluation function for Bayesian Optimization
def xgb_evaluate(n_estimators, learning_rate, max_depth, min_child_weight, subsample, colsample_bytree):
    params = {
        'n_estimators': int(n_estimators),
        'learning_rate': learning_rate,
        'max_depth': int(max_depth),
        'min_child_weight': int(min_child_weight),
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'objective': 'reg:squarederror'
    }
    model = XGBRegressor(**params)
    cv_score = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    return np.sqrt(-cv_score.mean())

# Define hyperparameter space
pbounds = {
    'n_estimators': (100, 1000),
    'learning_rate': (0.01, 0.3),
    'max_depth': (3, 10),
    'min_child_weight': (1, 6),
    'subsample': (0.7, 1.0),
    'colsample_bytree': (0.7, 1.0)
}

# Initialize Bayesian Optimizer
optimizer = BayesianOptimization(
    f=xgb_evaluate,
    pbounds=pbounds,
    random_state=42,
)

# Perform optimization
optimizer.maximize(init_points=10, n_iter=50)

# Output the best hyperparameters
best_params = optimizer.max['params']
best_params['n_estimators'] = int(best_params['n_estimators'])
best_params['max_depth'] = int(best_params['max_depth'])
best_params['min_child_weight'] = int(best_params['min_child_weight'])

# Train final model with best parameters
best_model = XGBRegressor(**best_params, objective='reg:squarederror')
best_model.fit(X_train, y_train)

# Model stacking/ensembling
level0 = [
    ('xgb', XGBRegressor(**best_params)),
    ('gbr', GradientBoostingRegressor(n_estimators=100))
]
level1 = Ridge()
stacking_model = StackingRegressor(estimators=level0, final_estimator=level1)
stacking_model.fit(X_train, y_train)

# Make predictions on the test set
test_data['SalePrice'] = stacking_model.predict(X_test)

today = datetime.today().strftime('%Y-%m-%d')

filename = f"submission_{today}.csv"
test_data[['Id', 'SalePrice']].to_csv(filename, index=False)

print(f"Predictions saved to {filename}")
