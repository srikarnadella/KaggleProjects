# KaggleProjects
https://www.kaggle.com/srikarnadella

## Titanic - Machine Learning from Disaster
Used an XGBoost model. Had to do a lot of data cleaning to prepare the dataset for the training since XGBoost does not take categorical data and the data was messy overall. My steps were first to remove the data I found irrelevant such as ticket number and name. I then mapped the genders to binary numbers as well as the embarked port. I then scaled age and fare cost to standardize them to limit the range of the data. This scored a 0.78468, which placed me at 2340 out of 15795 at the time of my writing. I believe I can move significantly up the leaderboard with a few key changes such as one-hot encoding the cabin and including that as well as optimizing the model by playing around with the parameters.


## House Prices - Advanced Regression Techniques
Given the size of this dataset (80 columns) there was significant data cleaning, preprocessing, and hyperparameter tuning to achieve a competitive score. I tried optimizing the hyper-parameters a lot to try and eek out a higher accuracy but was stalled out. Below are the details of two different versions of the code used, including their performance and key differences.

### Version 1 (traditional xgboost model main.py)
#### Approach
In this version, I performed the following steps:
* Data Loading: Loaded the train and test datasets into pandas Data Frames
* Feature Selection: Selected a comprehensive list of features relevant to predicting house prices.
* Data Preprocessing: 
  * Handled missing values using median for numerical features and most frequent for categorical features.
  * One-hot encoded categorical features to convert them into a numerical format suitable for the XGBoost model.
  * Standardized numerical features to ensure they are on a similar scale.

#### Model Training:
* Used an XGBoost regressor that had its hyperparameters tuned by RandomizedSearchCV.
* Model Evaluation: Assessed the model using cross-validation and mean squared error metrics.
* Prediction and Output: Predicted house prices on the test set and saved the output in a CSV file with a timestamp.

#### Performance
This version scored 0.13686 (aiming to get a 0), which placed me at 1619 out of 5029 on the leaderboard at the time of writing.

#### Changes and Improvements for the Future
* Implement K-means clustering in the beginning to try to find more relationships between the data
* Try reducing the amount of features to reduce over-fitting
* Find a better way of tuning the hyper-params rather than random sampling

### Version 2 (Stacking Model and Bayesian Optimization)
#### Approach
In this version, I made several changes to outperform the previous model primarily by implementing Bayesian Optimization.

* Enhanced Feature Selection: Included additional relevant features
* Optimized Data Preprocessing:
 * Improved handling of missing values.
 * Added feature engineering steps, such as extracting parts of the date and creating new interaction terms.
* Advanced Hyperparameter Tuning: Used a smaller hyperparameter search space for RandomizedSearchCV.
* Model Stacking to improve prediction quality

#### Performance
This version scored a 0.13871, placing it at 1871 out of 5029 on the leaderboard at the time of writing. 

#### Changes and Improvements for the Future
* Overall similar changes to the version 1 need to be made. Specifically a larger focus on reducing the features to prevent overfitting
* Implement K-means clustering in the beginning to try to find more relationships between the data
* Try reducing the amount of features to reduce over-fitting
