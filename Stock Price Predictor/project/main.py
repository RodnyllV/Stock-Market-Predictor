import numpy
import pandas
import matplotlib.pyplot as pyPlot
import seaborn
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = '0'

import tensorflow
import keras

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

import warnings
warnings.filterwarnings("ignore")

# Get data
teslaDF = pandas.read_csv("project/data/tesla.csv")
teslaDF = teslaDF.iloc[::-1]
teslaDF["date"] = pandas.to_datetime(teslaDF["date"])

predictionRange = teslaDF.loc[(teslaDF["date"] > datetime(2015, 1, 1)) & (teslaDF["date"] < datetime(2018, 1, 1))] 
closeData = teslaDF.filter(["close"])
dataSet = closeData.values
trainingDataLength = int(numpy.ceil(len(dataSet) * 0.95))

# Scale the data
scaler = MinMaxScaler(feature_range = (0, 1))
scaledData = scaler.fit_transform(dataSet)

# Prepare features and labels
trainingData = scaledData[0:trainingDataLength, :]
xTrain, yTrain = [], []

for i in range(60, len(trainingData)):
    xTrain.append(trainingData[i-60:i, 0])
    yTrain.append(trainingData[i, 0])

xTrain, yTrain = numpy.array(xTrain), numpy.array(yTrain)
xTrain = numpy.reshape(xTrain, (xTrain.shape[0], xTrain.shape[1], 1))

# Create and train the machine learning model
trainingModel = keras.models.Sequential()
trainingModel.add(keras.layers.LSTM(
    units = 64,
    return_sequences = True,
    input_shape = (xTrain.shape[1], 1)
))
trainingModel.add(keras.layers.LSTM(units = 64))
trainingModel.add(keras.layers.Dense(32))
trainingModel.add(keras.layers.Dropout(0.5))
trainingModel.add(keras.layers.Dense(1))

trainingModel.compile(optimizer = 'adam', loss = 'mean_squared_error')
history = trainingModel.fit(xTrain, yTrain, epochs = 10)

# Create testing data
testData = scaledData[trainingDataLength - 60:, :]
xTest, yTest = [], dataSet[trainingDataLength:, :]

for i in range(60, len(testData)):
    xTest.append(testData[i - 60:i, 0])

xTest = numpy.array(xTest)
xTest = numpy.reshape(xTest, (xTest.shape[0], xTest.shape[1], 1))

# Predict based off the testing data
predictions = scaler.inverse_transform(trainingModel.predict(xTest))

# Graph the predictions
train = teslaDF[:trainingDataLength]
test = teslaDF[trainingDataLength:]
test["predictions"] = predictions

pyPlot.figure(figsize = (10, 8))
pyPlot.plot(train['date'], train['close'])
pyPlot.plot(test['date'], test[['close', 'predictions']])
pyPlot.title("Tesla Stock Closing Price")
pyPlot.xlabel("Date")
pyPlot.ylabel("Close")
pyPlot.legend(["Train", "Test", "Predictions"])

pyPlot.show()
