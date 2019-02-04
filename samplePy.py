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
from sklearn.svm import LinearSVR
from sklearn import svm
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error as rmse
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error as mae
from mlxtend.regressor import StackingRegressor
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from itertools import permutations
from scipy.stats import skew
from scipy.special import boxcox1p

# Import feature tidying by the R implementation
R = robjects.r
R.source("sample2.R")
fullData = pandas2ri.ri2py_dataframe(R['fullData'])
testIDs = np.arange(1461, 2920)

# MAIN FUNCTION
def main():
    savePredictionByAllModels()

def runTuningXGB():
    data = getFinalFeatures()
    print("[Info] TUNING XGB - Starting...")
    tuningXGB = tuneXGB(data)
    print("[Info] TUNING XGB - Done.\n\n")
    return tuningXGB

def runTuningSVM():
    data = getFinalFeatures()
    print("[Info] TUNING SVM - Starting...")
    tuningSVM = tuneSVM(data)
    print("[Info] TUNING SVM - Done.")
    return tuningSVM

# Train simple models, ensemble and stacked. Save their prediction on files.
def savePredictionByAllModels():
    data = getFinalFeatures()

    lasso = getTrainedLasso(data)
    ridge = getTrainedRidge(data)
    xgb = getTrainedXGB(data)
    svm = getTrainedSVM(data)
    # ensemble = getTrainedEnsemble(data)
    # stacked  = getTrainedStacked(data)

    predLasso    = predictSalePrices(lasso, data)
    predRidge    = predictSalePrices(ridge, data)
    predXGB      = predictSalePrices(xgb, data)
    predSVM      = predictSalePrices(svm, data)
    # predStacked  = predictSalePrices(stacked, data)
    # predEnsemble = predictEnsemble(ensemble, data)

    savePredictionsOnFile(testIDs, predLasso,   createOutFilepath("Lasso"))
    savePredictionsOnFile(testIDs, predRidge,   createOutFilepath("Ridge"))
    savePredictionsOnFile(testIDs, predXGB,     createOutFilepath("XGB"))
    savePredictionsOnFile(testIDs, predSVM,     createOutFilepath("SVM"))
    # savePredictionsOnFile(testIDs, predEnsemble,createOutFilepath("Ensemble"))
    # savePredictionsOnFile(testIDs, predStacked, createOutFilepath("Stacked"))

    # predictions = (predLasso, predRidge, predXGB, predSVM, predEnsemble, predStacked)
    # models = (lasso, ridge, xgb, svm, ensemble, stacked)
    # return models

# OUTPUT WRITING
# The functions below are used to handle the output.
def createOutFilepath(modelName):
    prefix = "2019_02_04_ReTuned_"
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
    prices  = data.iloc[:, -1]
    return (predictors, prices)

# TRAINED MODELS
# The functions below return the trained models used in the final ensemble.
def getTrainedSVM(data):
    np.random.seed(1234)
    data = getTrainData(data)
    (predictors, prices) = getPredictorsAndPrices(data)
    print("[Info] Create SVM...")
    model = getTunedSVM()
    print("[Info] Train SVM...")
    model.fit(predictors, prices)
    print("[Info] SVM Trained.")
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
    metaModel = svm.SVR(kernel='rbf', gamma='scale')

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

def getEnsemble(data, weights):
    lasso = getTunedLasso(data)
    ridge = getTunedRidge(data)
    xgb = getTunedXGB(data)
    svm = getTunedSVM(data)

    ensemble = (lasso, ridge, xgb, svm)
    #weights  = (0.5, 0.5, 3.5, 5)

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
    return Lasso(alpha=0.0002)

def getTunedRidge():
    return Ridge(alpha=2.0, fit_intercept=True)

def getTunedSVM():
    return LinearSVR(C=0.00125, epsilon=0.1)

def getTunedXGB():
    # return xgb.XGBRegressor( colsample_bytree=0.4603, gamma=0.0468,
    #                          learning_rate=0.05, max_depth=3,
    #                          min_child_weight=1.7817, n_estimators=2200,
    #                          reg_alpha=0.4640, reg_lambda=0.8571,
    #                          subsample=0.5213, silent=1,
    #                          random_state =7, nthread = -1)
    return xgb.XGBRegressor( colsample_bytree=0.75, gamma=0.01, eta=0.005,
                             learning_rate=0.05, max_depth=4,
                             min_child_weight=2, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.8, silent=1,
                             random_state =7, nthread = -1)


# TUNING TEST
# The remaining functions are used to tuning the models, test their performance.
# The are WILD script and we cannot ensure their maintainance.
def tuneSVM(data):
    Cs = [0.0001, 0.001, 0.01, 0.1, 1, 5]
    epsilons = [0.01, 0.1]
    paramGrid = {'C': Cs, 'epsilon' : epsilons}
    model = LinearSVR(max_iter=100000)
    res = getTunedModel(model, paramGrid, data)
    printResultOfCVTuning(res)

def tuneXGB(data):
    nrounds = [750]
    min_child_weight = [2]
    gamma = [0.01]
    etas = [0.005, 0.0075, 0.01]
    max_depths = [4]
    colsample_bytrees = [0.65, 0.75, 0.95]
    subsamples = [0.7, 0.8, 0.9, 1]
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

def ensembleCV():
    allWeights = [0, 0.25, 0.5, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5, 4, 4.5, 5]
    perms = permutations(allWeights)
    permSize = len(perms)
    iters = 1

    data = getFinalFeatures()
    partitions = R.getKDataPartitions(data, iters)

    results = pd.DataFrame()
    for i in range(0, iters):
        cvTrain = pandas2ri.ri2py_dataframe(partitions[i][0])
        cvTestFull = pandas2ri.ri2py_dataframe(partitions[i][1])

        (cvTest, groundTruth) = getPredictorsAndPrices(cvTestFull)
        minRMSE = np.inf
        minWeights = None

        for currWeights in perms:
            (ensembleModel, _) = getEnsemble(cvTrain, currWeights)
            ensemblePreds = predictEnsemble(ensembleModel, cvTest)

            score = rmse(groundTruth, ensemblePreds, multioutput='uniform_average')
            if(score < minRMSE):
                minRMSE = score
                minWeights = currWeights

        (ensembleModel, _) = getEnsemble(cvTrain, minWeights)
        preds = predictEnsemble(ensembleModel, cvTest)
        res = pd.DataFrame()
        res['w0'] = minWeights[0]
        res['w1'] = minWeights[1]
        res['w2'] = minWeights[2]
        res['w3'] = minWeights[3]
        res['R2'] = r2_score(groundTruth, ensemblePreds, multioutput='uniform_average')
        res['RMSE'] = rmse(groundTruth, ensemblePreds, multioutput='uniform_average')
        res['MAE'] = mae(groundTruth, ensemblePreds, multioutput='uniform_average')

        results.append(res)
    results


def printResultOfCVTuning(result):
    # print("\nAll Iterations\n")
    # cv_results = result.cv_results_
    # for mean_score, params in zip(cv_results["mean_test_score"], cv_results["params"]):
    #     print(params, mean_score)
    print("\nBest parameters\n")
    print(result.best_params_)

def StolenaDataTidying():
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')

    train = train[~((train['GrLivArea'] > 4000) & (train['SalePrice'] < 300000))]
    all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'], test.loc[:,'MSSubClass':'SaleCondition']))

    # drop some features to avoid multicollinearity
    all_data.drop(['1stFlrSF', 'GarageArea', 'TotRmsAbvGrd'], axis=1, inplace=True)
    train["SalePrice"] = np.log1p(train["SalePrice"])
    numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
    skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
    skewed_feats = skewed_feats[skewed_feats > 0.65]
    skewed_feats = skewed_feats.index
    all_data[skewed_feats] = boxcox1p(all_data[skewed_feats], 0.15)
    all_data = pd.get_dummies(all_data)
    all_data = all_data.fillna(all_data.mean())
    return all_data


if __name__=="__main__":
    main()
