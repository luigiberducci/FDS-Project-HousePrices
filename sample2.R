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
library(Boruta) # NOT USEFUL ANYMORE
library(corrplot)
library(scales)
library(Rmisc)
library(ggrepel)
library(psych)
library(xgboost)
library(caret)
library(neuralnet) # NOT USEFUL ANYMORE

source("featureEngineering.R")
source("modelTesting.R")
options(warn=-1)

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

makeFinalPredictions <- function(){
    print("Cleaning dataset...")
    fullData <- getFinalFeatures(fullData)
    
    print("Training the optimal ensemble model...")
    # optimal ensemble model
    models <- c(getLassoModel, getRidgeModel, getXGBModel, getSVM)
    weights <- c(0.5, 0.5, 3.5, 5)
    
    ensemble <- createEnsembleModel(models, weights, fullData)
    ensemblePreds <- ensemblePredict(ensemble, fullData)
    
    print("Training XGB model...")
    # XGB
    xgb <- getXGBModel(fullData)
    xgbPreds <- predictSalePrices(xgb, fullData)
    
    print("Training SVM model...")
    # SVM
    svm <- getSVM(fullData)
    svmPreds <- predictSalePrices(svm, fullData)
    
    # Stacked regressors
    recipe <- list(getLassoModel, getRidgeModel, getXGBModel, getSVM)
    
    print("Training stacked regressor, variant A...")
    stackedA <- getStackedRegressor(fullData, recipe, getXGBModel, variant = "A")
    stackPredsA <- stackedA$predictions
    
    print("Training stacked regressor, variant B...")
    stackedB <- getStackedRegressor(fullData, recipe, getXGBModel, variant = "B")
    stackPredsB <- stackedB$predictions
    
    print("Averaging predictions...")
    # final predictions
    preds <- (2*ensemblePreds + xgbPreds + svmPreds + stackPredsA + stackPredsB)/6
    
    print("Writing CSV...")
    # write CSV
    savePredictionsOnFile(testIDs, preds, "out/final_predictions.csv")
    
    print("Done.")
}

# NOT USEFUL ANYMORE
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

# NOT USEFUL ANYMORE
test <- function(data){
    lasso <- getLassoModel(data)
    ridge <- getRidgeModel(data)
    xgb   <- getXGBModel(data)
    svm   <- getSVM(data)
    
    predictionsLasso <- predictSalePrices(lasso, data)
    predictionsRidge <- predictSalePrices(ridge, data)
    predictionsXGB   <- predictSalePrices(xgb, data)
    predictionsSVM   <- predictSalePrices(svm, data)

    res <- (2*predictionsLasso + 1.5*predictionsRidge + 1.5*predictionsXGB + 2*predictionsSVM)/7
    res
}

# NOT USEFUL ANYMORE
testCVonEnsemble <- function(data){
    data <- bootstrap(data)
    data <- importanceSelection(data, getLassoModel)

    models <- c(getLassoModel, getRidgeModel, getXGBModel)
    weights <- c(2, 1.5, 1.5)

    result <- iterateCrossValidationEnsembleModel(models, weights, data, 10)
    result
}

# NOT USEFUL ANYMORE
testConsistencyEnsemble <- function(data){
    data <- bootstrap(data)
    data <- importanceSelection(data, getLassoModel)

    lasso <- getLassoModel(data)
    ridge <- getRidgeModel(data)
    xgb   <- getXGBModel(data)
    predictionsLasso <- predictSalePrices(lasso, data)
    predictionsRidge <- predictSalePrices(ridge, data)
    predictionsXGB   <- predictSalePrices(xgb, data)

    manualRes <- (2*predictionsLasso + 1.5*predictionsRidge + 1.5*predictionsXGB)/5

    models <- c(getLassoModel, getRidgeModel, getXGBModel)
    weights <- c(2, 1.5, 1.5)
    ensemble <- createEnsembleModel(models, weights, data)
    ensembleRes <- ensemblePredict(ensemble, data)

    result <- data.frame(manual=manualRes, ensemble=ensembleRes)
    result
}

# NOT USEFUL ANYMORE
compareNewFeatures3 <- function(data){
    # By default, there is a new feature "totalSF"  
    data3 <- bootstrap(data, recentGarage=T)

    models <- c(getLassoModel, getRidgeModel, getXGBModel, getSVM)
    weights <- c(2, 1.5, 1.5, 2)

    cv3 <- iterateCrossValidationEnsembleModel(models, weights, data3, 5)
    cv3
}

# NOT USEFUL ANYMORE
compareNewFeatures2 <- function(data){
    # By default, there is a new feature "totalSF"  
    data2 <- bootstrap(data, carsXarea=T)

    models <- c(getLassoModel, getRidgeModel, getXGBModel, getSVM)
    weights <- c(2, 1.5, 1.5, 2)

    cv2 <- iterateCrossValidationEnsembleModel(models, weights, data2, 5)
    cv2
}

# NOT USEFUL ANYMORE
compareNewFeatures1 <- function(data){
    # By default, there is a new feature "totalSF"  
    data1 <- bootstrap(data, totBathRms=T)

    models <- c(getLassoModel, getRidgeModel, getXGBModel, getSVM)
    weights <- c(2, 1.5, 1.5, 2)

    cv1 <- iterateCrossValidationEnsembleModel(models, weights, data1, 5)
    cv1
}

library(gtools)

testEnsembleWeights <- function(data){
    print("Starting")
    print(Sys.time())

    models <- list(getSimpleLinearModel, getLassoModel, getRidgeModel, getENModel, getGradientBoostingModel, getSVM)
    weights <- c(1, 1, 1, 1, 1, 1)

    allWeights <- c(0, 0.25, 0.5, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5, 4, 4.5, 5)
    permutation <- permutations(length(allWeights), length(models), allWeights, repeats.allowed=T)
    numPermutations <- length(permutation[,1])
    numIterations <- 10

    cat(sprintf("[Info] Ensemble of %d models. Test %d weight permutations\n", length(models), length(permutation)))

    data <- getFinalFeatures(data)
    data <- getTrainData(data)

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
    
    print("Ending")
    print(Sys.time())

    finalRes
}

test5models <- function(data){
    data <- bootstrap(data, totBathRms=T, carsXarea=T, totalSF=T)
    data <- importanceSelection(data, getLassoModel)

    models <- c(getLassoModel, getRidgeModel, getENModel, getXGBModel, getSVM)
    weights <- c(5, 5, 5, 5, 5)

    ensemble <- createEnsembleModel(models, weights, data)
    pred <- ensemblePredict(ensemble, data)
    savePredictionsOnFile(testIDs, pred, "out/5models_with_new_feats_remove_multicoll.csv")
}

testOptimalWeightsAverage <- function(data){
    data <- getFinalFeatures(data)

    models <- c(getLassoModel, getRidgeModel, getXGBModel, getSVM)
    weights <- c(2.6, 1.7, 3.35, 2.25)
    weights <- c(1.425, 0.85, 2.45, 2.25)

    ensemble <- createEnsembleModel(models, weights, data)
    pred <- ensemblePredict(ensemble, data)
    savePredictionsOnFile(testIDs, pred, "out/2019_02_03_4models_optimal_weights.csv")
}

testOptimalWeightsMinimum <- function(data){
    data <- getFinalFeatures(data)

    models <- c(getLassoModel, getRidgeModel, getXGBModel, getSVM)
    weights <- c(0.5, 0.5, 3.5, 5)

    ensemble <- createEnsembleModel(models, weights, data)
    pred <- ensemblePredict(ensemble, data)
    savePredictionsOnFile(testIDs, pred, "out/2019_02_02_4models_min_optimal_weights.csv")
    pred
}

testOptimalWeightsMinimumMAE <- function(data){
    data <- getFinalFeatures(data)

    models <- c(getLassoModel, getRidgeModel, getGradientBoostingModel, getSVM)
    weights <- c(1.5, 0.5, 3.5, 4.5)

    ensemble <- createEnsembleModel(models, weights, data)
    pred <- ensemblePredict(ensemble, data)
    savePredictionsOnFile(testIDs, pred, "out/2019_02_03_4models_minMAE_optimal_weights.csv")
}

    
printf <- function(...) cat(sprintf(...))

printStackedTestInformation <- function(baseModels, metaModel, nIters){
    printf("[Info] Testing Stacked Regression with %d base models\n", length(baseModels))
    # printf("[Info] Base Models:")
    # printf("[Info] Meta-model: %s\n", as.character(metaModel))
    printf("[Info] Test: %d iteration of Cross-validation\n\n", nIters)
}

testStackedRegressor<- function(data, metaModelConstructor, baseModelList, nIters=10){
    # Debug
    printStackedTestInformation(baseModelList, metaModelConstructor, nIters)

    data <- getFinalFeatures(data)
    
    baseModels <- baseModelList
    metaModel <- metaModelConstructor
    variants <- c("A", "B")
    nIterations <- nIters

    resForBothVariants <- data.frame()
    for (var in variants){
        printf("[Info] Testing variant %s (%s)\n", var, Sys.time())
        stackedRecipe <- list( baseModelList = baseModels, 
                               metaModel = metaModel, 
                               variant = var)
        res <- iterateCrossValidationNTimes(getStackedRegressor, data, nIterations, TRUE, stackedRecipe)
        res$Variant <- var
        resForBothVariants <- rbind(resForBothVariants, res)
    }

    # Debug
    print(resForBothVariants)
    resForBothVariants
}

testStackedRegressorWt2BaseModels<- function(data, metaModel){
    dynamicDuo <- list(getLassoModel, getGradientBoostingModel)
    res <- testStackedRegressor(data, metaModel, dynamicDuo)
    res
}

testStackedRegressorWt3BaseModels<- function(data, metaModel){
    fantasticTrio <- list(getLassoModel, getRidgeModel, getGradientBoostingModel)
    res <- testStackedRegressor(data, metaModel, fantasticTrio)
    res
}

testStackedRegressorWt4BaseModels <- function(data, metaModel){
    theMagic4 <- list(getLassoModel, getRidgeModel, getGradientBoostingModel, getSVM)
    res <- testStackedRegressor(data, metaModel, theMagic4)
    res
}

testStackedRegressorWt5BaseModels <- function(data, metaModel){
    theBest5 <- list(getLassoModel, getRidgeModel, getENModel, getGradientBoostingModel, getSVM)
    res <- testStackedRegressor(data, metaModel, theBest5)
    res
}

testStackedRegressorWt6BaseModels <- function(data, metaModel){
    allOurPower <- list(getSimpleLinearModel, getLassoModel, getRidgeModel, getENModel, getGradientBoostingModel, getSVM)
    res <- testStackedRegressor(data, metaModel, allOurPower)
    res
}

testMyStackedRegressorWtLM <- function(data){
    # No good results: CV on 10 iterations
    # Variant A:
    #   R2   avg: 0.9211865 min: 0.8966896  max: 0.9333614
    #   RMSE avg: 0.1122499 min: 0.1004029  max: 0.1264718
    #   MAE  avg: 0.0766614 min: 0.07037704 max: 0.08501357
    # Variant B:
    #   R2   avg: 0.9215595  min: 0.9081677  max: 0.9381078
    #   RMSE avg: 0.1128418  min: 0.1009114  max: 0.1285855
    #   MAE  avg: 0.07738889 min: 0.07193925 max: 0.08547848
    res <- testStackedRegressorWt4BaseModels(data, getSimpleLinearModel)
    res
}

testMyStackedRegressorWtSVM <- function(data){
    outputDir <- "out"
    outputCSVName <- "2019_02_02_test_stacked_wt_svm"

    outName2 <- paste(outputCSVName, "2", sep="")
    outName3 <- paste(outputCSVName, "3", sep="")
    outName4 <- paste(outputCSVName, "4", sep="")
    outName5 <- paste(outputCSVName, "5", sep="")
    outName6 <- paste(outputCSVName, "6", sep="")

    outName2 <- paste(outName2, "csv", sep=".")
    outName3 <- paste(outName3, "csv", sep=".")
    outName4 <- paste(outName4, "csv", sep=".")
    outName5 <- paste(outName5, "csv", sep=".")
    outName6 <- paste(outName6, "csv", sep=".")

    outTest2 <- paste(outputDir, outName2, sep="/")
    outTest3 <- paste(outputDir, outName3, sep="/")
    outTest4 <- paste(outputDir, outName4, sep="/")
    outTest5 <- paste(outputDir, outName5, sep="/")
    outTest6 <- paste(outputDir, outName6, sep="/")


    res2bases <- testStackedRegressorWt2BaseModels(data, getSVM)
    write.csv(res2bases, outTest2, row.names=F)
    print(outTest2)

    res3bases <- testStackedRegressorWt3BaseModels(data, getSVM)
    write.csv(res3bases, outTest3, row.names=F)
    print(outTest3)

    res4bases <- testStackedRegressorWt4BaseModels(data, getSVM)
    write.csv(res4bases, outTest4, row.names=F)
    print(outTest4)
    print(Sys.time())

    res5bases <- testStackedRegressorWt5BaseModels(data, getSVM)
    write.csv(res5bases, outTest5, row.names=F)
    print(outTest5)
    print(Sys.time())

    res6bases <- testStackedRegressorWt6BaseModels(data, getSVM)
    write.csv(res6bases, outTest6, row.names=F)
    print(outTest6)
    print(Sys.time())
}

printStatisticsResultDF <- function(df){
    min  <- df %>% group_by(Variant) %>% summarise_all(min)  %>% as.data.frame
    mean <- df %>% group_by(Variant) %>% summarise_all(mean) %>% as.data.frame
    max  <- df %>% group_by(Variant) %>% summarise_all(max)  %>% as.data.frame

    min$Stat  <- "Min"
    mean$Stat <- "Mean"
    max$Stat  <- "Max"

    all <- rbind(min, mean, max)
    res <- data.frame(Variant=all$Variant, Stat=all$Stat, R2=all$R2, RMSE=all$RMSE, MAE=all$MAE)
    res[order(res$Variant),]
}

testTuneStackedRegressionSVM <- function(data){
    res <- testStackedRegressorWt2BaseModels(data, getSVM)
    res
}

testMyStackedRegressorWtRidge <- function(data){
    res <- testStackedRegressorWt4BaseModels(data, getRidgeModel)
    res
}

testMyStackedRegressorWtXGB <- function(data){
    res <- testStackedRegressorWt4BaseModels(data, getXGBModel)
    res
}

testNastedEnsemble <- function(data){
    data <- getFinalFeatures(data)
   
    # Ensemble
    models <- c(getLassoModel, getRidgeModel, getGradientBoostingModel, getSVM)
    weights <- c(0.5, 0.5, 3.5, 5)
    ensemble <- createEnsembleModel(models, weights, data)
    ensemblePred <- ensemblePredict(ensemble, data)
   
    # Stacked
    models <- c(getLassoModel, getRidgeModel, getENModel, getGradientBoostingModel)
    metamodel <- getSVM
    stacked <- getStackedRegressor(data, models, metamodel, "A")
    stackedPred <- stacked$predictions
    
    finalPred <- 0.5 * ensemblePred + 0.5 * stackedPred
    savePredictionsOnFile(testIDs, finalPred, "out/2019_02_02_average_ensamble_stacked.csv")
}

testStackedWithEN <- function(data){
    data <- getFinalFeatures(data)

    models <- c(getLassoModel, getRidgeModel, getENModel, getGradientBoostingModel)
    metamodel <- getSVM
    stacked <- getStackedRegressor(data, models, metamodel, "A")
    stackedPred <- stacked$predictions

    savePredictionsOnFile(testIDs, stackedPred, "out/2019_02_02_stacked_lassoridgeenxgb_svm.csv")
}

testBestStackedRegressor <- function(data){
    data <- getFinalFeatures(data)
    allOurPower <- list(getSimpleLinearModel, getLassoModel, getRidgeModel, getENModel, getGradientBoostingModel, getSVM)
    metamodel <- getSVM
    stacked <- getStackedRegressor(data, allOurPower, metamodel, "A")
    stackedPred <- stacked$predictions

    savePredictionsOnFile(testIDs, stackedPred, "out/2019_02_02_best_stacked_ever.csv")
}

theLastAttempt <- function(data){
    allOurPower <- list(getSimpleLinearModel, getLassoModel, getRidgeModel, getENModel, getGradientBoostingModel, getSVM)
    
    ensemble <- createEnsembleModel(models, weights, data)
}
