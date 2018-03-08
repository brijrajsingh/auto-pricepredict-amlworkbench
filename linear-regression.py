import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
import pickle
from matplotlib import pyplot as plt


# Use the Azure Machine Learning data preparation package
from azureml.dataprep import package

# This call will load the referenced package and return a DataFrame.
# If run in a PySpark environment, this call returns a
# Spark DataFrame. If not, it will return a Pandas DataFrame.
df = package.run('prep1.dprep', dataflow_idx=0,spark=False)

# Remove this line and add code that uses the DataFrame
print(df.head(10))

# Put the target (car price - price) in another DataFrame
columns = "num-of-doors fuel-type width height num-of-cylinders engine-type horsepower".split() # Declare the columns names
target = pd.DataFrame(df, columns=["price"])

X = pd.DataFrame(df,columns=columns)
y = target["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# default split is 75% for training and 25% for testing
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

lm = linear_model.LinearRegression()
model = lm.fit(X_train,y_train)

# print the intercept and coefficients
print(lm.intercept_)
print(lm.coef_)

# make predictions on the testing set
y_pred = lm.predict(X_test)
print(y_pred[0:5])

print("Score:", lm.score(X_test, y_test))

plt.scatter(y_test,y_pred)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.show()

#predict a set of real values
print('predicting against a set of values')
d = {'num-of-doors': [4], 'fuel-type': [1],'width':[68.9],'height':[55.5],'num-of-cylinders':[6],'engine-type':[0],'horsepower':[106]}
df = pd.DataFrame(data=d)
y_pred = lm.predict(df)
print(y_pred)

print ("Export the model to model.pkl")
f = open('./outputs/model.pkl', 'wb')
pickle.dump(lm, f)
f.close()