# KaggleProjects
https://www.kaggle.com/srikarnadella

## Titanic - Machine Learning from Disaster
Used an XGBoost model. Had to do a lot of data cleaning to prepare the dataset for the training since XGBoost does not take categorical data and the data was messy overall. My steps were first to remove the data I found irrelevant such as ticket number and name. I then mapped the genders to binary numbers as well as the embarked port. I then scaled age and fare cost to standardize them to limit the range of the data. This scored a 0.78468, which placed me at 2340 out of 15795 at the time of my writing. I believe I can move significantly up the leaderboard with a few key changes such as one-hot encoding the cabin and including that as well as optimizing the model by playing around with the parameters.


