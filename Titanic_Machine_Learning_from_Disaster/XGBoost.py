from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score

#Common generic dataset of flowers with data regarding petals and sepals and the length of width of each
iris = load_iris()

#rows by columns are samples by features
numSamples, numFeatures = iris.data.shape

#splitting the data in train and test with a 80% train data set size
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=0)

#train and test
#IMPORTANT NOTE; you need to convert the matrix to a DMAtrix not a numpy array
train = xgb.DMatrix(X_train, label=y_train)
test = xgb.DMatrix(X_test, label=y_test)

param = {
    'max_depth': 4, #First guess and then fidigit it
    'eta': 0.2,
    'objective': 'multi:softmax', #softmax means the most likely classification, soft prob gives the probabilties 
    'num_class': 3 #tells the number of classifications
    } 
epochs = 10 #number of iterations

model = xgb.train(param, train, epochs)

predictions = model.predict(test)

#returns accuracy score with 1 being perfect and 0 being not a single correct
output = accuracy_score(y_test, predictions)
print("output",output)