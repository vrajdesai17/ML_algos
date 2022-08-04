# IMPORTING THE LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# IMPORTING THE DATASET
dataset = pd.read_csv("../regression-models/Position_Salaries.csv")
x = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values
print(x)
print(y)
y = y.reshape(len(y),1) # For feature scaling , the StandardScalr accepts 2-D arrrays
print(y)

# FEATURE SCALING
from sklearn.preprocessing import StandardScalar
sc_x = StandardScalar()
sc_y = StandardScalar()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y)

# TRAINING THE SVR MODEL ON WHOLE DATASET
from sklearn.svm import SVR
regressor = SVR(kernel="rbf")
regressor.fit(x,y)

# PREDICTING THE NEW RESULT
sc_y.inverse_transform(regressor.predict(sc_x.transform([[6.5]])).reshape(-1,1))

# VISUALIZING THE SVR RESULT
plt.scatter(sc_x.inverse_transform(x),sc_y.inverse_transform(y),color="red")
plt.plot(sc_x.inverse_transform(x),sc_y.inverse_transform(regressor.predict(x)),color="blue")
plt.title("Truth or bluff")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()
