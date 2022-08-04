# IMPORTING THE LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# IMPORTING THE DATASET
dataset = pd.read_csv("/regression-models/Position_Salaries.csv")
x = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

# TRAINING THE SIMPLE LINEAR REGRESSION MODEL ON DATASET
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x,y) 

# TRAINING THE POLYNOMIAL LINEAR REGRESSION MODEL ON DATASET
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
x_poly = poly_reg.fit_transform(x)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly,y)

# VISUALIZING THE SIMPLE LINEAR REGRESSION MODEL
plt.scatter(x,y,color="red")
plt.plot(x,lin_reg.predict(x),color="blue")
plt.title("Truth or bluff")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()

# VISUALIZING THE POLYNOMIAL LINEAR REGRESSION MODEL
plt.scatter(x,y,color="red")
plt.plot(x,lin_reg_2.predict(poly_reg.fit_transform(x)),color="blue")
plt.title("Truth or bluff")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()

# PREDICTING A NEW RESULT WITH SIMPLE LINEAR REGRESSION MODEL
lin_reg.predict([[6.5]])

# PREDICTING A NEW RESULT WITH POLYNOMIAL LINEAR REGRESSION MODEL
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))