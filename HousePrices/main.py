import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.compose import ColumnTransformer
import numpy as np
from scipy.stats import uniform, randint
from datetime import datetime
import time

# Load datasets
# Read the training and testing data into pandas DataFrames to allow us to modify them easily
traindf = pd.read_csv('train.csv')
testdf = pd.read_csv('test.csv')

# List of features to be used in the model based on typical influencers of housing prices
features = ['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street', 'Alley', 'LotShape',
            'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1',
            'Condition2', 'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt',
            'YearRemodAdd', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
            'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 
            'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 
            'BsmtUnfSF', 'TotalBsmtSF', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical',
            '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath',
            'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 
            'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType', 'GarageYrBlt', 
            'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond', 'PavedDrive', 
            'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',
            'PoolQC', 'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType', 'SaleCondition']

# Filter out features that are not in the dataset
# Ensuring all listed features are present in the training data
features = [feature for feature in features if feature in traindf.columns]

# Splitting the dataset into features (X) and target variable (y)
X_train = traindf[features]
y_train = traindf['SalePrice']
X_test = testdf[features]

# Preprocessing pipeline for numerical features
# This will handle missing values and standardize numerical features
numerical_features = X_train.select_dtypes(include=['int64', 'float64']).columns
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),  # Replace missing values with median (avoids swaying the data)
    ('scaler', StandardScaler())  # Standardize features by removing the mean and scaling to unit variance
])

# Preprocessing pipeline for categorical features
# This will handle missing values and one-hot encode categorical features
categorical_features = X_train.select_dtypes(include=['object']).columns
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Replace missing values with the most frequent value (avoids swaying the data)
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  # Convert categorical features into a one-hot encoded format
])

# Combine preprocessing pipelines into a single ColumnTransformer
# This applies the appropriate transformations to each column in the dataset
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),  # Apply numerical transformer to numerical features
        ('cat', categorical_transformer, categorical_features)  # Apply categorical transformer to categorical features
    ]
)

# Create the model pipeline
# This pipeline first applies the preprocessor and then fits the XGBRegressor
model = Pipeline(steps=[
    ('preprocessor', preprocessor),  # Apply the preprocessing steps
    ('regressor', XGBRegressor(objective='reg:squarederror', n_jobs=-1))  # Use XGBoost Regressor for prediction
])

# Define hyperparameter search space for RandomizedSearchCV
param_dist = {
    'regressor__n_estimators': randint(100, 1000),  # Number of boosted trees to fit
    'regressor__learning_rate': uniform(0.01, 0.1),  # Step size shrinkage used in update to prevent overfitting
    'regressor__max_depth': randint(3, 10),  # Maximum depth of a tree
    'regressor__min_child_weight': randint(1, 6),  # Minimum sum of instance weight needed in a child
    'regressor__subsample': uniform(0.7, 0.3),  # Subsample ratio of the training instances
    'regressor__colsample_bytree': uniform(0.7, 0.3)  # Subsample ratio of columns when constructing each tree
}

# Hyperparameter tuning using RandomizedSearchCV
# This searches across param_dist to find the best hyperparameters
random_search = RandomizedSearchCV(
    model, param_distributions=param_dist, n_iter=100, cv=5, 
    scoring='neg_mean_squared_error', verbose=1, n_jobs=-1, random_state=42
)
random_search.fit(X_train, y_train)

# Output best parameters and score from RandomizedSearchCV
print("Best parameters:", random_search.best_params_)
print("Best score:", np.sqrt(-random_search.best_score_))

# Train final model with best hyperparameters
best_model = random_search.best_estimator_
best_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = best_model.predict(X_test)

# Get current date and time to create a unique filename
today = datetime.today()
formatted_date = today.strftime("%m-%d")
t = time.localtime()
current_time = time.strftime("%H:%M", t)

# Save the predictions to a CSV file with a timestamped filename
output_data = pd.DataFrame({'Id': testdf['Id'], 'SalePrice': y_pred})
filename = f"output_{formatted_date}_{current_time}.csv"
output_data.to_csv(filename, index=False)
