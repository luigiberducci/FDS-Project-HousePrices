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

main <- function(data){
    # Clean features
    data <- bootstrap(data)
    # Features selection
    # Prediction
    pred <- runPrediction(data)
    # Write output
    savePredictionsOnFile(testIDs, pred, "out/2019_02_01_mod.csv")
}

bootstrap <- function(data){
    data <- removeOutliers(data)
    data <- handleSkewness(data)
    
    futureDeletionNames <- getFactorFields(data)
    data <- handleNA(data)
    data <- appendDummyVariables(data)
    data[futureDeletionNames] <- NULL
    
    data <- addFeatureTotalSF(data)

    data <- replaceRemainingNAwtMean(data)
    data <- removeMulticollinearFeatures(data)
}

runPrediction <- function(data){
    testData <- getTestData(data)
    lasso <- getLassoModel(data)
    ridge <- getRidgeModel(data)
    xgb   <- getGradientBoostingModel(data)
    predictionslasso <- expm1(predict(lasso, testData))
    predictionsridge <- expm1(predict(ridge, testData))
    predictionsxgb   <- expm1(predict(xgb, testData))
    res <- (2*predictionslasso+1.5*predictionsridge+1.5*predictionsxgb)/5
    res
}
