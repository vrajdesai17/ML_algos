# IMPORTING THE LIBRARIES
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# IMPORTING THE DATASET
dataset = pd.read_csv("./regression-models/50_Startups.csv")
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values
print(x)
print(y)
y = y.reshape(len(y),1)
print(y)

# FEATURE SCALING 
from sklearn.preprocessing import StandardScalar
sc_x = StandardScalar()
sc_y = StandardScalar()
x = sc_x.fit_transform(x)
y = sc_y.fit_tranform(y)

# TRAINING THE SVR MODEL ON WHOLE DATASET
from sklearn.svm import SVR
regressor = SVR(kernel="rbf")
regressor.fit(x,y)

# PREDICTING THE NEW RESULT 
sc_y.inverse_transform(regressor.predict(sc_x.fit_transform([[6.5]])).reshape(-1,1))


# VISIUALIZING THE SVR MODEL RESULTS
plt.scatter(sc_x.inverse_transform(x),sc_y.inverse_tranform(y),color="red")
plt.plot(sc_x.inverse_transform(x),regressor.predict(sc_y.inverse_tranform(y).reshape(-1,1)),color="blue")
plt.title("Salary vs Experience(Training set)")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()