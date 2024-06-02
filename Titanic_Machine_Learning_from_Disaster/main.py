from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

train_path = r"H:\CodingProjects\KaggleProjects\Titanic_Machine_Learning_from_Disaster\train.csv"
test_path =  r"H:\CodingProjects\KaggleProjects\Titanic_Machine_Learning_from_Disaster\test.csv"
traindf = pd.read_csv(train_path)
testdf = pd.read_csv(test_path)


print(traindf.head())

#Data Cleaning:

#Drop the name column as that is irrelevant
traindf = traindf.drop('Name', axis=1)

#Changing the word male and female to numbers
maleFemaleDictionary = {'male': 1, 'female': 0}
traindf['Sex'] = traindf['Sex'].map(maleFemaleDictionary)

traindf = traindf.drop('Cabin', axis=1)
#Replacing the values that say other to NaN
#traindf['Cabin'] = traindf['Cabin'].replace(['Other'], np.nan)

#Removed ticket because that is irrelevant to the outcome
traindf = traindf.drop('Ticket', axis=1)

#Standardizing age and ticket fare
scale = StandardScaler()
traindf[['Age', 'Fare']] = scale.fit_transform(traindf[['Age', 'Fare']].values)

#Cleaning Embarked column and replacing words
embarkedDictionary = {'C': 0, 'Q': 1, 'S': 2}
traindf['Embarked'] = traindf['Embarked'].map(embarkedDictionary)

X_train = traindf[['Pclass', 'Sex', 'Age','SibSp','Parch','Fare','Embarked']]
y_train = traindf[['Survived']]

#One hot encode the cabin col
#X_train = pd.get_dummies(X_train, columns=['Cabin'])

#Repeating for the test dataset
testdf = testdf.drop('Name', axis=1)
maleFemaleDictionary = {'male': 1, 'female': 0}
testdf['Sex'] = testdf['Sex'].map(maleFemaleDictionary)
testdf = testdf.drop('Cabin', axis=1)
#testdf['Cabin'] = testdf['Cabin'].replace(['Other'], np.nan)
testdf = testdf.drop('Ticket', axis=1)
scale = StandardScaler()
testdf[['Age', 'Fare']] = scale.fit_transform(testdf[['Age', 'Fare']].values)
embarkedDictionary = {'C': 0, 'Q': 1, 'S': 2}
testdf['Embarked'] = testdf['Embarked'].map(embarkedDictionary)
X_test = testdf[['Pclass', 'Sex', 'Age','SibSp','Parch','Fare','Embarked']]
#X_test = pd.get_dummies(X_test, columns=['Cabin'])


print(traindf.head())


train = xgb.DMatrix(X_train, label=y_train,enable_categorical=True)

X_test = xgb.DMatrix(X_test)

param = {
    'max_depth': 4, 
    'eta': 0.2,
    'objective': 'multi:softmax', #Going with softmax as we want the most likely
    'num_class': 2 
    } 
epochs = 10 #number of iterations

model = xgb.train(param, train, epochs)

predictions = model.predict(X_test)


output_data = pd.DataFrame({'Survived': predictions})
output_data.to_excel("output.xlsx", index=False)
