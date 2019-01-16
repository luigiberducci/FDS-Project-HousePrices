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

def trainModelAndTestItOnTestData(train, test):
    X, y = featureSelection(train, test)
    lr = linear_model.LinearRegression()
    model = lr.fit(X, y)
    observations = test.select_dtypes(include=[np.number]).drop('Id', axis=1).interpolate()
    predictions = model.predict(observations)
    finalPredictions = np.exp(predictions) #because we put log of SalePrice
    return finalPredictions

def testModelInLoopOnPartialTestData(train, test, times):
    X, y = featureSelection(train, test)
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

def featureSelection(train, test):
    # Step 1
    train = removeOutliers(train)
    # Step 2 - one-hot encoding for 'street' feature
    train, test = transformCategoricalToNumerical(train, test)
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

def transformCategoricalToNumerical(train, test):
    # Encoding Street Condition
    train['enc_street'] = pd.get_dummies(train.Street, drop_first=True)
    test['enc_street']  = pd.get_dummies(test.Street, drop_first=True)

    # Encoding Sale Condition
    train['enc_condition'] = train.SaleCondition.apply(encodeSaleCond)
    test['enc_condition']  = test.SaleCondition.apply(encodeSaleCond)

    return train, test

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
