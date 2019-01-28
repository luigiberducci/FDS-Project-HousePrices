# Authors:  Angelo Di Mambro, Emanuele Giona, Luigi Berducci
# Date:     January 2019
# Purpose:  Create a main script which merge the individual work done.
#           Remember that the script is shared among the students on Github.
#           Then before commit the update, invoke "git stash", "git pull" and "git stash pop".

# Libraries
library(ggplot2)
library(plyr)
library(dplyr)
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

fullData <- rbind(train, test)

# Features' blocks (NEVER USED, TODO: remove it if unnecessary)
location <- c("MSSubClass", "MSZoning", "Street", "Alley", "Neighborhood", "Condition1", "Condition2")
lot <- c("LotFrontage", "LotArea", "LotShape", "LandContour", "LotConfig", "LandSlope")
misc <- c("Utilities", "BldgType", "HouseStyle", "Heating", "HeatingQC", "CentralAir", "Electrical", "Fireplaces", "FireplaceQu", "MiscFeature", "MiscVal")
outside = c("PavedDrive", "Fence", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "ScreenPorch") 
kitchen = c("KitchenAbvGr", "KitchenQual")
bathrooms = c("BsmtFullBath", "BsmtHalfBath", "FullBath", "HalfBath")
general = c("OverallQual", "OverallCond", "Functional")
rooms = c("BedroomAbvGr", "TotRmsAbvGrd")

# Main functions
#performs feature engineering and only keeps relevant features
bootstrap <- function(data){
    data <- featureEngineering(data)
    data
}

bootstrap2 <- function(data){
    #removing outliers
    trainData <- getTrainData(data)
    testData <- getTestData(data)

    trainData <- trainData %>%
        filter(SalePrice < 600000) %>%                              #4 houses
        filter(LotFrontage <= 200) %>%                              #2 houses
        filter(LotArea <= 100000) %>%                               #4 houses
        filter(Fireplaces < 3) %>%                                  #5 houses
        filter(!(MiscFeature %in% c("TenC", "Othr", "Gar2"))) %>%   #4 houses
        filter(MiscVal < 5000)                                      #2 houses
    data <- rbind(trainData, testData)
    
    data <- handleLocations2(data)
    data <- handleLot(data)
    data <- handleMisc2(data)
    
    data <- handleSaleBsmtAndYears2(data)
    
    data <- handleGarage(data)
    data <- handleRooms(data)
    data <- handleOutside2(data)

    data <- addFeatureBathrooms(data)
    data <- addFeatureCarsXArea(data)
    data <- addFeatureRecentGarage(data)
    data <- addFeatureRecentType(data)

    # from factor to ordinal 0 (poor), 1 (low), 2 (medium), 3 (rich)
    # Note: seems that neigh conversion doesn't improve the score
    # data <- convertNeighboroodToClasses(data)

    #removing multicollinear features
    base <- c("X1stFlrSF","GarageArea", "TotRmsAbvGrd")
    bathrooms <- c("BsmtFullBath", "BsmtHalfBath", "FullBath", "HalfBath")
    garage <- c("GarageCars", "GarageType", "GarageYrBlt")
    multicollinear <- c(base, bathrooms, garage)
    data <- data[!colnames(data) %in% multicollinear]
    
    #log prices
    data$SalePrice[!is.na(data$SalePrice)] <- log(data$SalePrice[!is.na(data$SalePrice)])
    SKEWCORRECTION <<- TRUE

    #checking skewness and applying BoxCox transformation to skewed features
    factors <- getFactorFields(data)
    numericFeats <- names(which(sapply(data, is.numeric)))
    numericFeats <- numericFeats[numericFeats != "SalePrice"]
    numericData <- data[numericFeats]
    skewness <- showSkewness(numericData) %>% filter(!is.nan(Skewness))
    skewFeats <- skewness$Feature[abs(skewness$Skewness) > 0.65]
    skewFeats <- setdiff(skewFeats, factors)
    transformation <- preProcess(data[skewFeats], method = "BoxCox")
    processed <- predict(transformation, data[skewFeats])
    data[skewFeats] <- processed
    
    #adding dummy variables
    data <- appendDummyVariables(data)
    data <- removeFactors(data)
    
    data
}

# here instructions to automatically perform bootstrapping, test on real test set and saving predictions to file
