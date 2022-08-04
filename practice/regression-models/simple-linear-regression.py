# IMPORTING THE LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# IMPORTING THE DATASET
dataset = pd.read_csv("/regression-models/Salary_Data.csv")
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

# SPLITTING THE DATASET INTO TRAINING SET AND TEST SET
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

# TRAINING THE SIMPLE LINEAR REGRESSION MODEL ON TRAINING SET
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train) 

# PREDICTING THE TEST SET RESULTS
y_pred = regressor.predict(x_test)

# VISIUALIZING THE TRAINING SET RESULTS
plt.scatter(x_train,y_train,color="red")
plt.plot(x_train,regressor.predict(x_train),color="blue")
plt.title("Salary vs Experience(Training set)")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()

# VISUALIZING THE TEST SET RESULTS
plt.scatter(x_test,y_test,color="red")
plt.plot(x_test,y_pred,color="blue")
plt.title("Salary vs Experience(Test set)")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()
