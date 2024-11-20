import numpy
import pandas
import requests
import matplotlib.pyplot as pyPlot
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = '0'

import tensorflow
import keras

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

import warnings
warnings.filterwarnings("ignore")

# Ask the user for the company stock
ticker = input("Enter a stock to predict: ")

# Set up the URL
# API KEY - 5FDGVWUH1D95L529
url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&outputsize=full&apikey=5FDGVWUH1D95L529'
req = requests.get(url)
dataJSON = req.json()

# Create a dataframe from the alphavantage URL
stockData = pandas.DataFrame(dataJSON.get("Time Series (Daily)")).transpose()
stockData.index.name = "date"
stockData.columns = ["open", "high", "low", "close", "volume"]
stockData = stockData.sort_index()
stockData.reset_index(inplace = True)

# Get data
teslaDF = pandas.read_csv("project/data/tesla.csv")
# teslaDF = teslaDF.iloc[::-1]
stockData["date"] = pandas.to_datetime(stockData["date"])

predictionRange = stockData.loc[(stockData["date"] > datetime(2020, 1, 1)) & (stockData["date"] < datetime(2025, 1, 1))] 
closeData = stockData.filter(["close"])
dataSet = closeData.values
trainingDataLength = int(numpy.ceil(len(dataSet) * 0.95))

for column in stockData.columns:
    if column != "date":
        stockData[column] = pandas.to_numeric(stockData[column])

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
train = stockData[:trainingDataLength]
test = stockData[trainingDataLength:]
test["predictions"] = predictions

pyPlot.figure(figsize = (10, 8))
pyPlot.plot(train["date"], train["close"])
pyPlot.plot(test["date"], test[["close", "predictions"]])
pyPlot.title(ticker + " Stock Closing Price w/ Predictions")
pyPlot.xlabel("Date")
pyPlot.ylabel("$USD")
pyPlot.legend(["Train", "Test", "Predictions"])

pyPlot.show()
