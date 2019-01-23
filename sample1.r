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
outside = c("PavedDrive", "Fence", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "ScreenPorch", "PoolArea", "PoolQC")
garage = c("GarageType", "GarageYrBlt", "GarageFinish", "GarageCars", "GarageArea", "GarageQual", "GarageCond")
rooms = c("BedroomAbvGr", "TotRmsAbvGrd")
kitchen = c("KitchenAbvGr", "KitchenQual")
bathrooms = c("BsmtFullBath", "BsmtHalfBath", "FullBath", "HalfBath")
general = c("OverallQual", "OverallCond", "Functional")

luigi  <- c(general, bathrooms, kitchen, rooms, garage, outside, 'SalePrice')
angelo <- c("YearBuilt","YearRemodAdd","MoSold","YrSold","SaleType","SaleCondition","RoofStyle","RoofMatl","Exterior1st","Exterior2nd","MasVnrType","MasVnrArea","ExterQual","ExterCond","Foundation","BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinSF1","BsmtFinType2","BsmtFinSF2","BsmtUnfSF","TotalBsmtSF","LowQualFinSF","GrLivArea","SalePrice")

# List of Ordinal variables conversion
AccessType  <- c("None" = 0, "Grvl" = 1, "Pave" = 2)
LotShape    <- c("IR3" = 0, "IR2" = 1, "IR1" = 2, "Reg" = 3)
LandSlope   <- c("Sev" = 0, "Mod" = 1, "Gtl" = 2)
Utilities   <- c("None" = 0, "ELO" = 1, "NoSeWa" = 2, "NoSewr" = 3, "AllPub" = 4)
CentralAir  <- c("N" = 0, "Y" = 1)
Qualities   <- c('None'=0, 'Po' = 1, 'Fa' = 2, 'TA' = 3, 'Gd' = 4, 'Ex' = 5)
Exposure    <- c('None'=0, 'No'=1, 'Mn'=2, 'Av'=3, 'Gd'=4)
FinType     <- c('None'=0, 'Unf'=1, 'LwQ'=2, 'Rec'=3, 'BLQ'=4, 'ALQ'=5, 'GLQ'=6)
Masonry     <- c('None'=0, 'BrkCmn'=0, 'BrkFace'=1, 'Stone'=2)
PavedDrive  <- c('Y'=1, 'N'=0, 'P'=0)
Functional  <- c('Typ'=7, 'Min1'=6, 'Min2'=5, 'Mod'=4, 'Maj1'=3, 'Maj2'=2, 'Sev'=1, 'Sal'=0)
GarageFinish <- c('Fin'=3, 'RFn'=2, 'Unf'=1, 'Miss'=0)

# Helper Functions
#returns the most important features, estimated via the Boruta technique; can be set up to only work on selected features (default: all) and to meet a certain importance threshold (default: 0)
applyBoruta <- function(data, features = NULL, threshold = 0){
  set.seed(12345)
  if(!is.null(features)){
    data <- data %>%
      select(., SalePrice, features) %>%
      filter(., !is.na(.$SalePrice))
  }

  borutaImp <- Boruta(data$SalePrice~., data = data, doTrace = 0)
  goodFeatures <- getSelectedAttributes(borutaImp)
  stats <- attStats(borutaImp) %>%
    filter(., .$decision == "Confirmed")

  stats$feature <- goodFeatures
  stats <- stats %>%
    filter(.$meanImp >= threshold) %>%
    select(., feature, meanImp) %>%
    arrange(.$meanImp)
}

#returns the mode of a collection of values, ignoring NAs
getMode <- function(values){
  uniques <- unique(values)
  uniques <- uniques[!is.na(uniques)]
  maxFreqID <- which.max(tabulate(match(values,uniques)))
  uniques[maxFreqID]
}

#returns values as factors, filling NAs with the mode or with the specified argument
getFactors <- function(values, replaceNA = "NA"){
  if(!is.factor(values))
  {
    values[is.na(values)] <- ifelse(replaceNA == "NA", getMode(values), replaceNA)
    values <- as.factor(values)
  }
  values
}

#returns values as integers, enconding them via the given dictionary, filling NAs with the mode or with the specified argument
getIntegers <- function(values, dictionary, replaceNA = "NA"){
  if(!is.integer(values))
  {
    values[is.na(values)] <- ifelse(replaceNA == "NA", getMode(values), replaceNA)
    values <- as.integer(dictionary[values])
  }
  values
}

#returns average frontage given a certain neighborhood
avgFrontage <- function(neighborhood){
  as.integer(mean(fullData$LotFrontage[fullData$Neighborhood == neighborhood], na.rm = T))
}

#substitues all NAs with the average frontage of the same neighborhood
getValidFrontages <- function(data){
  if(0 < length(which(is.na(data$LotFrontage))))
  {
    for(i in 1:nrow(data))
      if(is.na(data$LotFrontage[i]))
        data$LotFrontage[i] <- avgFrontage(data$Neighborhood[i])
  }
  data
}

#fixes incoherent MiscFeature and MiscVal values (MiscFeature = NA <-> MiscVal = 0)
getValidMiscFeaturesAndVal <- function(data){
  if(0 < length(which((is.na(data$MiscFeature) & data$MiscVal > 0) | (!is.na(data$MiscFeature) & data$MiscVal == 0))))
  {
    for(i in 1:nrow(data)){
      if(is.na(data$MiscFeature[i]) & data$MiscVal[i] > 0)
        data$MiscVal[i] <- 0
      else if(!is.na(data$MiscFeature[i]) & data$MiscVal[i] == 0)
        data$MiscFeature[i] <- "None"
    }
  }
  data
}

# Main functions
featureEngineering <- function(data){
    #Emanuele
    data <- handleLocations(data) 
    data <- handleLot(data) 
    data <- handleMisc(data) 
    #Angelo
    data <- handleSaleBsmtAndYears(data)
    #Luigi
    data <- handleGarage(data)
    data <- handleOutside(data)
    data <- handleRooms(data)
    data <- addFeatureBathrooms(data)
    data <- addFeatureRecentGarage(data)
    data <- addFeatureCarsXArea(data)

    data <- appendDummyVariables(data)
    data
}

removeFactors <- function(data){
    factNames <- getFactorFields(data)
    data <- data[!(names(data) %in% factNames)]
    data
}

appendDummyVariables <- function(data){
    fact <- getFactorData(data)
    dummies <- as.data.frame(model.matrix(~.-1, fact))
    data <- cbind(data, dummies)
    data
}

handleRooms <- function(data){
    data$KitchenQual <- getIntegers(data$KitchenQual, Qualities)
    data$Functional <- getIntegers(data$Functional, Functional, "Typ")
    data$KitchenAbvGr[is.na(data$KitchenAbvGr)] <- 0
    data$FullBath[is.na(data$FullBath)] <- 0
    data$HalfBath[is.na(data$HalfBath)] <- 0
    data$BsmtFullBath[is.na(data$BsmtFullBath)] <- 0
    data$BsmtHalfBath[is.na(data$BsmtHalfBath)] <- 0
    data
}

handleGarage <- function(data){
    #TODO Check consistency among Garage features
    data$GarageFinish <- getIntegers(data$GarageFinish, GarageFinish, "Miss" )
    data$GarageQual <- getIntegers(data$GarageQual, Qualities, "None" )
    data$GarageCond <- getIntegers(data$GarageCond, Qualities, "None" )
    data$GarageCars[is.na(data$GarageCars)] <- 0
    data$GarageYrBlt[is.na(data$GarageYrBlt)] <- 0
    data$GarageArea[is.na(data$GarageArea)] <- 0
    data$GarageType[!(data$GarageType == 'BuiltIn' | data$GarageType=='Attchd')] <- 'OT'
    data$GarageType[data$GarageType == 'BuiltIn' | data$GarageType=='Attchd'] <- 'BA'
    data$GarageType <- getFactors(data$GarageType, "OT")
    data
}

handleOutside <- function(data){
    #TODO Pool Consistency
    data$PavedDrive <- getIntegers(data$PavedDrive, PavedDrive, "N")
    data$Fence <- getFactors(data$Fence, "None")
    data$PoolQC <- getIntegers(data$PoolQC, Qualities, "None")
    data$PoolArea[is.na(data$PoolArea)] <- 0
    data
}

handleSaleBsmtAndYears <- function(data){
    data$MoSold         <- getFactors(data$MoSold)
    data$SaleType       <- getFactors(data$SaleType)
    data$SaleCondition  <- getFactors(data$SaleCondition)
    data$RoofStyle      <- getFactors(data$RoofStyle)
    data$RoofMatl       <- getFactors(data$RoofMatl)
    data$Exterior1st    <- getFactors(data$Exterior1st)
    data$Exterior2nd    <- getFactors(data$Exterior2nd)
    data$Foundation     <- getFactors(data$Foundation)
    data$ExterQual      <- getIntegers(data$ExterQual,Qualities)
    data$ExterCond      <- getIntegers(data$ExterCond,Qualities)
    data$MasVnrType     <- getIntegers(data$MasVnrType,Masonry, "None")
    data$BsmtQual       <- getIntegers(data$BsmtQual,Qualities, "None")
    data$BsmtCond       <- getIntegers(data$BsmtCond,Qualities, "None")
    data$BsmtExposure   <- getIntegers(data$BsmtExposure,Exposure, "None")
    data$BsmtFinType1   <- getIntegers(data$BsmtFinType1,FinType, "None")
    data$BsmtFinType2   <- getIntegers(data$BsmtFinType2,FinType, "None")
    data$BsmtFinSF1[is.na(data$BsmtFinSF1)] <-0
    data$BsmtFinSF2[is.na(data$BsmtFinSF2)] <-0
    data$BsmtUnfSF[is.na(data$BsmtUnfSF)] <-0
    data$TotalBsmtSF[is.na(data$TotalBsmtSF)] <-0
    data$MasVnrArea[is.na(data$MasVnrArea)] <-0
    data
}

handleLocations <- function(data){
    data$MSSubClass     <- getFactors(data$MSSubClass)
    data$MSZoning       <- getFactors(data$MSZoning)
    data$Street         <- getIntegers(data$Street, AccessType)
    data$Alley          <- getFactors(data$Alley, "None")
    data$Neighborhood   <- getFactors(data$Neighborhood)
    data$Condition1     <- getFactors(data$Condition1)
    data$Condition2     <- getFactors(data$Condition2)
    data
}

handleLot <- function(data){
    data                <- getValidFrontages(data)
    data$LotShape       <- getIntegers(data$LotShape, LotShape)
    data$LandContour    <- getFactors(data$LandContour)
    data$LotConfig      <- getFactors(data$LotConfig)
    data$LandSlope      <- getIntegers(data$LandSlope, LandSlope)
    data
}

handleMisc <- function(data){
    data$Utilities      <- getIntegers(data$Utilities, Utilities, "None")
    data$BldgType       <- getFactors(data$BldgType)
    data$HouseStyle     <- getFactors(data$HouseStyle)
    data$Heating        <- getFactors(data$Heating)
    data$HeatingQC      <- getIntegers(data$HeatingQC, Qualities, 'None')
    data$CentralAir     <- getIntegers(data$CentralAir, CentralAir, 'N')
    data$Electrical     <- getFactors(data$Electrical)
    data$FireplaceQu    <- getIntegers(data$FireplaceQu, Qualities, "None")
    data                <- getValidMiscFeaturesAndVal(data)
    data$MiscFeature    <- getFactors(data$MiscFeature, "None")
    data
}

addFeatureBathrooms <- function(data){
    prices <- data$SalePrice
    data$SalePrice <- NULL
    data$TotBathRms <- data$BsmtFullBath + 0.5*data$BsmtHalfBath + data$FullBath + 0.5*data$HalfBath
    data$SalePrice <- prices
    data
}

addFeatureRecentGarage <- function(data) {
    prices <- data$SalePrice
    data$SalePrice <- NULL
    data$RecentGarage[data$GarageYrBlt < 2000] <- 0
    data$RecentGarage[data$GarageYrBlt >= 2000] <- 1
    data$RecentGarage <- as.factor(data$RecentGarage)
    data$SalePrice <- prices
    data
}

addFeatureCarsXArea <- function(data) {
    prices <- data$SalePrice
    data$SalePrice <- NULL
    data$GarageCarsTimesArea <- data$GarageCars * data$GarageArea
    data$SalePrice <- prices
    data
}

addFeatureTypeXRecent <- function(data) {
    prices <- data$SalePrice
    data$SalePrice <- NULL
    data$GarageRecentType <- data$RecentGarage * data$GarageType
    data$SalePrice <- prices
    data
}

getFactorData <- function(data) {
    newData <- data[, getFactorFields(data)]
    newData
}

getFactorFields <- function(data) {
    newData <- which(sapply(data, is.factor))
    names(newData)
}
