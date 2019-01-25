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

# here instructions to automatically perform bootstrapping, test on real test set and saving predictions to file
