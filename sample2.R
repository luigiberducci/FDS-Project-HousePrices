# Authors:  Angelo Di Mambro, Emanuele Giona, Luigi Berducci
# Date:     January 2019
# Purpose:  Create a main script which merge the individual work done.
#           Remember that the script is shared among the students on Github.
#           Then before commit the update, invoke "git stash", "git pull" and "git stash pop".

# Libraries
library(ggplot2)
library(plyr)
library(dplyr)
library(tidyr)
library(reshape2)
library(Boruta)
library(corrplot)
library(scales)
library(Rmisc)
library(ggrepel)
library(psych)
library(xgboost)
library(caret)
library(neuralnet)

source("featureEngineering.R")
source("modelTesting.R")

# Dataset
train   <- read.csv("data/train.csv", stringsAsFactor=FALSE)
test    <- read.csv("data/test.csv", stringsAsFactor=FALSE)

# Preliminary data handling
testIDs  <- test$Id     # Save the Ids for submission
train$Id <- NULL        # and remove them in the dataset
test$Id  <- NULL
test$SalePrice <- NA    # Test hasn't any SalePrice, then set it as NA
`%!in%` = Negate(`%in%`)

fullData <- rbind(train, test)

getFinalFeatures <- function(data){
    data <- bootstrap(data)
    data <- importanceSelection(data, getLassoModel)
    data
}

bootstrap <- function(data, totBathRms=T, carsXarea=T, recentGarage=F, totalSF=T){
    data <- removeOutliers(data)
    data <- handleSkewness(data)
    
    futureDeletionNames <- getFactorFields(data)
    data <- handleNA(data)
    data <- appendDummyVariables(data)
    data[futureDeletionNames] <- NULL
    
    data <- addNewFeatures(data, totBathRms, carsXarea, recentGarage, totalSF)

    data <- replaceRemainingNAwtMean(data)
    data <- removeMulticollinearFeatures(data)
}

testCorrectnessRefactoring <- function(data){
    # Clean features
    data <- bootstrap(data)
    # Features selection
    relevantFeats <- importanceSelection(data, getLassoModel, 10)
    # Prediction
    pred <- test(relevantFeats)
    # Write output
    savePredictionsOnFile(testIDs, pred, "out/2019_02_01_1feat.csv")
}

test <- function(data){
    lasso <- getLassoModel(data)
    ridge <- getRidgeModel(data)
    xgb   <- getGradientBoostingModel(data)
    svm   <- getSVM(data)
    
    predictionsLasso <- predictSalePrices(lasso, data)
    predictionsRidge <- predictSalePrices(ridge, data)
    predictionsXGB   <- predictSalePrices(xgb, data)
    predictionsSVM   <- predictSalePrices(svm, data)

    res <- (2*predictionsLasso + 1.5*predictionsRidge + 1.5*predictionsXGB + 2*predictionsSVM)/7
    res
}


testCVonEnsemble <- function(data){
    data <- bootstrap(data)
    data <- importanceSelection(data, getLassoModel)

    models <- c(getLassoModel, getRidgeModel, getGradientBoostingModel)
    weights <- c(2, 1.5, 1.5)

    result <- iterateCrossValidationEnsembleModel(models, weights, data, 10)
    result
}

testConsistencyEnsemble <- function(data){
    data <- bootstrap(data)
    data <- importanceSelection(data, getLassoModel)

    lasso <- getLassoModel(data)
    ridge <- getRidgeModel(data)
    xgb   <- getGradientBoostingModel(data)
    predictionsLasso <- predictSalePrices(lasso, data)
    predictionsRidge <- predictSalePrices(ridge, data)
    predictionsXGB   <- predictSalePrices(xgb, data)

    manualRes <- (2*predictionsLasso + 1.5*predictionsRidge + 1.5*predictionsXGB)/5

    models <- c(getLassoModel, getRidgeModel, getGradientBoostingModel)
    weights <- c(2, 1.5, 1.5)
    ensemble <- createEnsembleModel(models, weights, data)
    ensembleRes <- ensemblePredict(ensemble, data)

    result <- data.frame(manual=manualRes, ensemble=ensembleRes)
    result
}

compareNewFeatures3 <- function(data){
    # By default, there is a new feature "totalSF"  
    data3 <- bootstrap(data, recentGarage=T)

    models <- c(getLassoModel, getRidgeModel, getGradientBoostingModel, getSVM)
    weights <- c(2, 1.5, 1.5, 2)

    cv3 <- iterateCrossValidationEnsembleModel(models, weights, data3, 5)
    cv3
}

compareNewFeatures2 <- function(data){
    # By default, there is a new feature "totalSF"  
    data2 <- bootstrap(data, carsXarea=T)

    models <- c(getLassoModel, getRidgeModel, getGradientBoostingModel, getSVM)
    weights <- c(2, 1.5, 1.5, 2)

    cv2 <- iterateCrossValidationEnsembleModel(models, weights, data2, 5)
    cv2
}

compareNewFeatures1 <- function(data){
    # By default, there is a new feature "totalSF"  
    data1 <- bootstrap(data, totBathRms=T)

    models <- c(getLassoModel, getRidgeModel, getGradientBoostingModel, getSVM)
    weights <- c(2, 1.5, 1.5, 2)

    cv1 <- iterateCrossValidationEnsembleModel(models, weights, data1, 5)
    cv1
}

library(gtools)

testEnsembleWeights <- function(data){
    allWeights <- c(0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6)
    permutation <- permutations(10, 4, allWeights, repeats.allowed=T)
    numPermutations <- length(permutation[,1])
    numIterations <- 10

    data <- bootstrap(data)
    data <- importanceSelection(data, getLassoModel)
    data <- getTrainData(data)

    models <- c(getLassoModel, getRidgeModel, getGradientBoostingModel, getSVM)
    weights <- c(1, 1, 1, 1)
   

    partitions <- getKDataPartitions(data, numIterations)
    finalRes <- data.frame()
    for(i in 1:numIterations){
        cat(sprintf("[Info] Iter=%d\n", i))
        partition <- partitions[[i]]

        ensemble <- createEnsembleModel(models, weights, partition$trainData)
        groundTruth <- partition$testData$SalePrice
        partition$testData$SalePrice <- NA
        minRMSE <- 1
        for(j in 1:numPermutations){
            if(j%%1000==0)
                cat(sprintf("\t[Info] Perm=%d\n", j))

            weights <- permutation[j,]
            ensemble$weights <- weights

            pred <- ensemblePredict(ensemble, partition$testData, checkSkew=F)
            rmse <- RMSE(pred, groundTruth)
            if(rmse<minRMSE){
                minWeights <- weights
                minRMSE <- rmse
            }
        }
        ensemble$weights <- minWeights
        bestPred <- ensemblePredict(ensemble, partition$testData, checkSkew=F)
        res   <- data.frame(w1 = minWeights[1],
                            w2 = minWeights[2],
                            w3 = minWeights[3],
                            w4 = minWeights[4],
                            R2 = R2(bestPred, groundTruth),
                            RMSE = RMSE(bestPred, groundTruth),
                            MAE = MAE(bestPred, groundTruth))
        finalRes <- rbind(finalRes, res)
        str(finalRes)
    }
    finalRes
}

test5models <- function(data){
    data <- bootstrap(data, totBathRms=T, carsXarea=T, totalSF=T)
    data <- importanceSelection(data, getLassoModel)

    models <- c(getLassoModel, getRidgeModel, getENModel, getGradientBoostingModel, getSVM)
    weights <- c(5, 5, 5, 5, 5)

    ensemble <- createEnsembleModel(models, weights, data)
    pred <- ensemblePredict(ensemble, data)
    savePredictionsOnFile(testIDs, pred, "out/5models_with_new_feats_remove_multicoll.csv")
}

testOptimalWeightsAverage <- function(data){
    data <- bootstrap(data, totBathRms=T, carsXarea=T, totalSF=T)
    data <- importanceSelection(data, getLassoModel)

    models <- c(getLassoModel, getRidgeModel, getGradientBoostingModel, getSVM)
    weights <- c(2.6, 1.7, 3.35, 2.25)

    ensemble <- createEnsembleModel(models, weights, data)
    pred <- ensemblePredict(ensemble, data)
    savePredictionsOnFile(testIDs, pred, "out/2019_02_02_4models_optimal_weights.csv")
}

testOptimalWeightsMinimum <- function(data){
    data <- bootstrap(data, totBathRms=T, carsXarea=T, totalSF=T)
    data <- importanceSelection(data, getLassoModel)

    models <- c(getLassoModel, getRidgeModel, getGradientBoostingModel, getSVM)
    weights <- c(0.5, 0.5, 3.5, 5)

    ensemble <- createEnsembleModel(models, weights, data)
    pred <- ensemblePredict(ensemble, data)
    savePredictionsOnFile(testIDs, pred, "out/2019_02_02_4models_min_optimal_weights.csv")
}

testStackedRegressorWt4BaseModels <- function(data, metaModelConstructor){
    printf <- function(...) cat(sprintf(...))
    data <- getFinalFeatures(data)
    
    theMagic4 <- c(getLassoModel, getRidgeModel, getGradientBoostingModel, getSVM)
    metaModel <- metaModelConstructor
    variants <- c("A", "B")
    nIterations <- 1

    resForBothVariants <- data.frame()
    for (var in variants){
        printf("[Info] Testing variant %s\n", var)
        stackedFormula <- list(baseModelList = theMagic4, 
                               metaModel = metaModel, 
                               variant = var)
        res <- iterateCrossValidationNTimes(getStackedRegressor, data, nIterations, TRUE, stackedFormula)
        res$Variant <- var
        resForBothVariants <- rbind(resForBothVariants, res)
    }
    resForBothVariants
}

testMyStackedRegressorWtLM <- function(data){
    res <- testStackedRegressorWt4BaseModels(data, getSimpleLinearModel)
    res
}

testMyStackedRegressorWtSVM <- function(data){
    res <- testStackedRegressorWt4BaseModels(data, getSVM)
    res
}
