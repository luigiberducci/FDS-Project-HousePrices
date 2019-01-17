import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
import os
import sys

DATA_DIR       = "data"
TRAIN_FILENAME = "train.csv"
TEST_FILENAME  = "test.csv"

SUBMISSION_DIR = "out"
SUBMISSION_FILENAME = "submission.csv"

def main(argv=[]):
    train = getTrainDataframe()
    test  = getTestDataframe()

    # Test N times on random part of training data
    N = 10
    scores = testModelInLoopOnPartialTestData(train, test, N)
    printResults(scores)

    # Test model trained in all training data and return predictions on test data
    predictions = trainModelAndTestItOnTestData(train, test)
    submission  = packSubmissionCSV(test, predictions)

    # Write submission file
    writeSubmissionCSVFile(submission)

def writeSubmissionCSVFile(submission):
    outputFilepath = os.path.join(SUBMISSION_DIR, SUBMISSION_FILENAME)
    submission.to_csv(outputFilepath, index=False)

def packSubmissionCSV(test, predictions):
    submission = pd.DataFrame()
    submission['Id'] = test.Id
    submission['SalePrice'] = predictions
    return submission

def printResults(scores):
    resultString = ""
    for i, scr in enumerate(scores):
        resultString += "  #Iteration:\t{}\t|\tR2: {}\n".format(i, scr)
    resultString += "Mean R2:\t{}".format(np.mean(scores))
    print(resultString)

def trainModelAndTestItOnTestData(train, test):
    X, y = featureSelection(train)
    lr = linear_model.LinearRegression()
    model = lr.fit(X, y)

    test  = transformCategoricalToNumerical(test)
    observations = test.select_dtypes(include=[np.number]).drop('Id', axis=1).interpolate()

    predictions = model.predict(observations)
    finalPredictions = np.exp(predictions) #because we put log of SalePrice
    return finalPredictions

def testModelInLoopOnPartialTestData(train, test, times):
    X, y = featureSelection(train)
    scores = []
    for it in range(0, times):
        R2 = trainModelAndEvaluateIt(X, y)
        scores.append(R2)
    return scores

def trainModelAndEvaluateIt(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    lr = linear_model.LinearRegression()
    model = lr.fit(X_train, y_train)
    return model.score(X_test, y_test)

def featureSelection(train):
    # Step 1
    train = removeOutliers(train)
    # Step 2 - one-hot encoding for 'street' feature
    train = transformCategoricalToNumerical(train)

    # Step 3 - interpolate missing values
    train = interpolateMissingValues(train)
    # Step 4 - log saleprice to reduce skewness
    target = np.log(train.SalePrice)

    y = target
    X = train.drop(['SalePrice', 'Id'], axis=1)
    return X, y

def interpolateMissingValues(train):
    interp = train.select_dtypes(include=[np.number]).interpolate()
    return interp.dropna()

def transformCategoricalToNumerical(dataframe):
    # Encoding Street Condition
    dataframe['enc_street'] = pd.get_dummies(dataframe.Street, drop_first=True)
    # Encoding Sale Condition
    dataframe['enc_condition'] = dataframe.SaleCondition.apply(encodeSaleCond)
    return dataframe

def encodeSaleCond(x):
    enc = 0
    if x=="Partial":
        enc = 1
    return enc

def getNumericalFeatures(train):
    numerical = train.select_dtypes(include=[np.number])
    return numerical

def getCategoricalFeatures(train):
    categorical = train.select_dtypes(exclude=[np.number])
    return categorical

def removeOutliers(train):
    train = train[ train['GarageArea'] < 1200 ]
    train = train[ train['GrLivArea'] < 4500 ]
    return train

def getTrainDataframe():
    filepath = os.path.join(DATA_DIR, TRAIN_FILENAME)
    return pd.read_csv(filepath)

def getTestDataframe():
    filepath = os.path.join(DATA_DIR, TEST_FILENAME)
    return pd.read_csv(filepath)

if __name__=="__main__":
    main(sys.argv)
