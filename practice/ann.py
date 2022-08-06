# IMPORTING THE LIBRARIES
import numpy as np
import pandas as pd
import tensorflow as tf

# IMPORTING THE DATASET
dataset = pd.read_csv("data.csv")
x = dataset.iloc[:,3:].values
y = dataset.iloc[:,-1].values

# STEP 1 : DATA-PREPROCESSING

# ENCODING THE CATEGORICAL DATA
# SPLITTING THE DATASET INTO TRAINING AND TEST SET
# FEATURE SCALING

# STEP 2 : BUILDING THE ANN

# INITIALIZING THE ANN
ann = tf.keras.models.Sequential()

# INPUT LAYER AND FIRST HIDDEN LAYER
ann.add(tf.keras.layers.Dense(units=6,activation="relu"))

# SECOND HIDDEN LAYER
ann.add(tf.keras.layers.Dense(units=6,activation="relu"))

# OUTPUT LAYER
ann.add(tf.keras.layers.Dense(units=1,activation="sigmoid"))

# STEP 3 : TRAINING THE ANN 

# COMPILING THE ANN
ann.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])

# TRAINING THE ANN
ann.fit(x_train,y_train,batch_size=32,epochs=100)


