# Authors:  Angelo Di Mambro, Emanuele Giona, Luigi Berducci
# Date:     January 2019
# Purpose:  The same approach proposed in sample2.R but using Python Language
#
#

# Libraries
import os
import numpy as np
import pandas as pd
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from sklearn import svm
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from mlxtend.regressor import StackingRegressor
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

# Import feature tidying by the R implementation
R = robjects.r
R.source("sample2.R")
fullData = pandas2ri.ri2py_dataframe(R['fullData'])
testIDs = np.arange(1461, 2920)

# Train simple models, ensemble and stacked. Save their prediction on files.
def main():
    data = getFinalFeatures()

    lasso = getTrainedLasso(data)
    ridge = getTrainedRidge(data)
    xgb = getTrainedXGB(data)
    svm = getTrainedSVM(data)
    ensemble = getTrainedEnsemble(data)
    stacked  = getTrainedStacked(data)

    predLasso    = predictSalePrices(lasso, data)
    predRidge    = predictSalePrices(ridge, data)
    predXGB      = predictSalePrices(xgb, data)
    predSVM      = predictSalePrices(svm, data)
    predStacked  = predictSalePrices(stacked, data)
    predEnsemble = predictEnsemble(ensemble, data)

    savePredictionsOnFile(testIDs, predLasso,   createOutFilepath("Lasso"))
    savePredictionsOnFile(testIDs, predRidge,   createOutFilepath("Ridge"))
    savePredictionsOnFile(testIDs, predXGB,     createOutFilepath("XGB"))
    savePredictionsOnFile(testIDs, predSVM,     createOutFilepath("SVM"))
    savePredictionsOnFile(testIDs, predEnsemble,createOutFilepath("Ensemble"))
    savePredictionsOnFile(testIDs, predStacked, createOutFilepath("Stacked"))

    predictions = (predLasso, predRidge, predXGB, predSVM, predEnsemble, predStacked)
    models = (lasso, ridge, xgb, svm, ensemble, stacked)
    return models

# OUTPUT WRITING
# The functions below are used to handle the output.
def createOutFilepath(modelName):
    prefix = "2019_02_04_tuned_"
    outDir = "out"
    extens = "csv"
    name = prefix + modelName.upper() + "." + extens
    return os.path.join(outDir, name)

def savePredictionsOnFile(testIDs, pred, outFile):
    df = pd.DataFrame()
    df['Id'] = testIDs
    df['SalePrice'] = pred
    outputContent = df.to_csv(index=False)
    with open(outFile, 'w') as out:
        out.write(outputContent)

# DATA HANDLING
# The functions below are used to handle data.
# For instance:
#   1) there is a function to extract the final features (cleaned and selected),
#   2) there are functions which return train dat or test data
#   3) there is also a function to the dataset splitted into predictors and prices.
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

# TRAINED MODELS
# The functions below return the trained models used in the final ensemble.
def getTrainedSVM(data):
    np.random.seed(1234)
    data = getTrainData(data)
    (predictors, prices) = getPredictorsAndPrices(data)
    model = getTunedSVM()
    model.fit(predictors, prices)
    return model

def getTrainedLasso(data):
    np.random.seed(1234)
    data = getTrainData(data)
    (predictors, prices) = getPredictorsAndPrices(data)
    model = getTunedLasso()
    model.fit(predictors, prices)
    return model

def getTrainedRidge(data):
    np.random.seed(1234)
    data = getTrainData(data)
    (predictors, prices) = getPredictorsAndPrices(data)
    model = getTunedRidge()
    model.fit(predictors, prices)
    return model

def getTrainedXGB(data):
    np.random.seed(1234)
    data = getTrainData(data)
    (predictors, prices) = getPredictorsAndPrices(data)
    model = getTunedXGB()
    model.fit(predictors, prices)
    return model

def getTrainedStacked(data):
    np.random.seed(1234)
    data = getTrainData(data)
    (predictors, prices) = getPredictorsAndPrices(data)
    lasso = getTunedLasso()
    ridge = getTunedRidge()
    XGB = getTunedXGB()
    SVM = getTunedSVM()
    metaModel = svm.SVR(kernel='rbf')

    stacked = StackingRegressor(regressors=[lasso, ridge, XGB, SVM],
                                meta_regressor=metaModel)
    stacked.fit(predictors, prices)
    return stacked

def getTrainedEnsemble(data):
    lasso = getTrainedLasso(data)
    ridge = getTrainedRidge(data)
    XGB   = getTrainedXGB(data)
    SVM   = getTrainedSVM(data)

    ensemble = (lasso, ridge, XGB, SVM)
    weights  = (0.5, 0.5, 3.5, 5)
    return (ensemble, weights)

# PREDICTIONS
# The functions below are used to predict sale price.
# Basically, there are 2 main function "predictSalePrices" and "predictEnsemble".

# Given a model, use it to predict SalePrice.
#
# Parameters:
# ===========
#   -`model`:   trained model built by sklearn
#   -`data`:    full dataset (train+test) composed only by cleaned features
def predictSalePrices(model, data):
    data = getTestData(data)
    data = data.iloc[:, :-1]
    pred = model.predict(data)
    return np.expm1(pred)

# Given an ensemble model, use it to predict SalePrice.
#
# Parameters:
# ===========
#   -`ensemble`: ensemble model built using "getTrainedEnsemble" composed of
#                a list of trained base `models` and a list of `weights`
#   -`data`:     full dataset (train+test) composed only by cleaned features
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

# DEFINITIVE TUNED BASE MODELS
# The functions below return the tuned models.
# Such models are NOT trained, they are just configured according to the best parameters.
def getTunedLasso():
    # return Lasso(alpha=0.0004)
    return Lasso(alpha=0.0004)

def getTunedRidge():
    return Ridge(alpha=7.6, fit_intercept=True)

def getTunedSVM():
    return svm.SVR(C=15, gamma=1e-06)

def getTunedXGB():
    # return xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.75, max_depth = 3)
    return xgb.XGBRegressor( colsample_bytree=0.4603, gamma=0.0468,
                             learning_rate=0.05, max_depth=3,
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)

# TUNING TEST
# The remaining functions are used to tuning the models, test their performance.
# The are WILD script and we cannot ensure their maintainance.
def tuneSVM(data):
    Cs = [0.1, 1, 7.5, 10, 12.5, 15]
    gammas = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1]
    paramGrid = {'C': Cs, 'gamma' : gammas}
    model = svm.SVR()
    res = getTunedModel(model, paramGrid, data)
    printResultOfCVTuning(res)

def tuneXGB(data):
    nrounds = [750]
    min_child_weight = [2]
    gamma = [0.01]
    etas = [0.01,0.005,0.001]
    max_depths = [4,6,8]
    colsample_bytrees = [0,1,10]
    subsamples = [0,0.2,0.4,0.6]
    paramGrid = {'nrounds': nrounds, 'min_child_weight': min_child_weight,
                 'gamma' : gamma, 'eta':etas, 'max_depth': max_depths,
                 'colsample_bytree':colsample_bytrees, 'subsample': subsamples}
    model = xgb.XGBRegressor()
    res = getTunedModel(model, paramGrid, data)
    printResultOfCVTuning(res)

def getTunedModel(model, paramGrid, data):
    data = getTrainData(data)
    nFolds = 10
    search = GridSearchCV(model, paramGrid, cv=nFolds)
    (predictors, prices) = getPredictorsAndPrices(data)
    search.fit(predictors, prices)
    return search

def printResultOfCVTuning(result):
    print("\nAll Iterations\n")
    cv_results = result.cv_results_
    for mean_score, params in zip(cv_results["mean_test_score"], cv_results["params"]):
        print(params, mean_score)
    print("\nBest parameters\n")
    print(result.best_params_)


