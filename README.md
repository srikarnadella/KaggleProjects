# KaggleProjects
https://www.kaggle.com/srikarnadella

## Titanic - Machine Learning from Disaster
Used an XGBoost model. Had to do a lot of data cleaning to prepare the dataset for the training since XGBoost does not take categorical data and the data was messy overall. My steps were first to remove the data I found irrelevant such as ticket number and name. I then mapped the genders to binary numbers as well as the embarked port. I then scaled age and fare cost to standardize them to limit the range of the data. This scored a 0.78468, which placed me at 2340 out of 15795 at the time of my writing. I believe I can move significantly up the leaderboard with a few key changes such as one-hot encoding the cabin and including that as well as optimizing the model by playing around with the parameters.


## Titanic - Machine Learning from Disaster
Given the size of this dataset (80 columns) there was significant data cleaning, preprocessing, and hyperparameter tuning to achieve a competitive score. I tried optimizing the hyper-parameters a lot to try and eek out a higher accuracy but was stalled out. Below are the details of two different versions of the code used, including their performance and key differences.

### Version 1
#### Approach
In this version, I performed the following steps:

Data Loading: Loaded the train and test datasets.
Feature Selection: Selected a comprehensive list of features relevant to predicting house prices.
Data Preprocessing:
Handled missing values using median for numerical features and most frequent for categorical features.
One-hot encoded categorical features to convert them into a numerical format suitable for the XGBoost model.
Standardized numerical features to ensure they are on a similar scale.
Model Training:
Used an XGBoost regressor with default parameters.
Model Evaluation:
Assessed the model using cross-validation and mean squared error metrics.
Prediction and Output:
Predicted house prices on the test set and saved the output in a CSV file with a timestamp.
Performance
This version scored 0.12784, which placed me at 2340 out of 15795 on the leaderboard at the time of writing.
