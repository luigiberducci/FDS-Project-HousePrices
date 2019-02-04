# Authors:  Angelo Di Mambro, Emanuele Giona, Luigi Berducci
# Date:     January 2019
# Purpose:  The same approach proposed in sample2.R but using Python Language
#
#

# Libraries
import numpy as np
import pandas as pd
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from sklearn import svm
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
import xgboost as xgb

R = robjects.r
R.source("sample2.R")
fullData = pandas2ri.ri2py_dataframe(R['fullData'])
testIDs = np.arange(1461, 2920)

def main():
    data = getFinalFeatures()
    lasso = getLassoModel(data)
    ridge = getRidgeModel(data)
    xgb = getXGBModel(data)
    svm = getSVM(data)
    ensemble = getEnsemble(data)

    predLasso = predictSalePrices(lasso, data)
    predRidge = predictSalePrices(ridge, data)
    predXGB   = predictSalePrices(xgb, data)
    predSVM   = predictSalePrices(svm, data)
    predEnsemble = predictEnsemble(ensemble, data)

    savePredictionsOnFile(testIDs, predLasso,   "out/2019_02_04_LASSO.csv")
    savePredictionsOnFile(testIDs, predRidge,   "out/2019_02_04_RIDGE.csv")
    savePredictionsOnFile(testIDs, predXGB,     "out/2019_02_04_XGB.csv")
    savePredictionsOnFile(testIDs, predSVM,     "out/2019_02_04_SVM.csv")
    savePredictionsOnFile(testIDs, predEnsemble,"out/2019_02_04_ENSEMBLE.csv")

    return(predLasso, predRidge, predXGB, predSVM, predEnsemble)

def savePredictionsOnFile(testIDs, pred, outFile):
    df = pd.DataFrame()
    df['Id'] = testIDs
    df['SalePrice'] = pred
    outputContent = df.to_csv(index=False)
    with open(outFile, 'w') as out:
        out.write(outputContent)

def predictSalePrices(model, data):
    data = getTestData(data)
    data = data.iloc[:, :-1]
    pred = model.predict(data)
    return np.expm1(pred)

def getFinalFeatures():
    data = R['fullData']
    data = R.getFinalFeatures(data)
    return pandas2ri.ri2py_dataframe(data)

def getTrainData(data):
    data = data[data.SalePrice.notnull()]
    return data

def getTestData(data):
    data = data[data.SalePrice.isnull()]
    return data

def getPredictorsAndPrices(data):
    predictors = data.iloc[:, :-1]
    prices  = data['SalePrice']
    return (predictors, prices)

def getSVM(data):
    np.random.seed(1234)
    data = getTrainData(data)
    (predictors, prices) = getPredictorsAndPrices(data)
    model = svm.SVR()
    model.fit(predictors, prices)
    return model

def getLassoModel(data):
    np.random.seed(1234)
    data = getTrainData(data)
    (predictors, prices) = getPredictorsAndPrices(data)
    model = Lasso(alpha=1)
    model.fit(predictors, prices)
    return model

def getRidgeModel(data):
    np.random.seed(1234)
    data = getTrainData(data)
    (predictors, prices) = getPredictorsAndPrices(data)
    model = Ridge(alpha=0.0)
    model.fit(predictors, prices)
    return model

def getXGBModel(data):
    np.random.seed(1234)
    data = getTrainData(data)
    (predictors, prices) = getPredictorsAndPrices(data)
    model = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.75, max_depth = 3)
    model.fit(predictors, prices)
    return model

def getEnsemble(data):
    lasso = getLassoModel(data)
    ridge = getRidgeModel(data)
    xgb = getXGBModel(data)
    svm = getSVM(data)

    ensemble = (lasso, ridge, xgb, svm)
    weights  = (0.5, 0.5, 3.5, 5)
    return (ensemble, weights)

def predictEnsemble(ensemble, data):
    models, weights = ensemble
    allPred   = getDFofIndividualPredictions(models, data)
    finalPred = mergeIndividualPredictionWtWeights(allPred, weights)
    return finalPred

def getDFofIndividualPredictions(models, data):
    allPred = pd.DataFrame()
    for i, model in enumerate(models):
        pred = predictSalePrices(model, data)
        allPred["Pred{}".format(i)] = pred
    return allPred

def mergeIndividualPredictionWtWeights(pred, weights):
    numberOfPred = pred.shape[0]
    finalPred = np.zeros(numberOfPred)
    for i, w in enumerate(weights):
        predAsArray = np.array(pred["Pred{}".format(i)])
        finalPred += w*predAsArray
    return finalPred





