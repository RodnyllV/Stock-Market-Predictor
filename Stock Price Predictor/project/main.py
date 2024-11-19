import numpy
import pandas
import matplotlib.pyplot as pyPlot
import seaborn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import metrics

import warnings
warnings.filterwarnings("ignore")

teslaDataFrame = pandas.read_csv("project/data/tesla.csv")
teslaStockDateSplit = teslaDataFrame["Date"].str.split('/', expand = True)

teslaStockDateSplit["Day"] = teslaStockDateSplit[1].asType("int")
teslaStockDateSplit["Month"] = teslaStockDateSplit[0].asType("int")
teslaStockDateSplit["Year"] = teslaStockDateSplit[2].asType("int")

"""
graphs = { # specifies the y axis
    "open": "$USD",
    "close": "$USD",
    "high": "$USD",
    "low": "$USD",
    "volume": "SHARES"
}
"""

graphs = ["open", "close", "high", "low", "volume"]

_, ax = pyPlot.subplots(figsize = (20, 10))

for subCt, column in enumerate(graphs):
    pyPlot.subplot(2, 3, subCt + 1)
    seaborn.boxplot(teslaDataFrame[column], ax = ax)

pyPlot.show()
