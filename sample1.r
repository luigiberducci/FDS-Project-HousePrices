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
source("featureSelection.R")
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
    data <- removeOutliers(data) 
    data <- encodeFeatures(data)
    data <- addNewFeatures(data)
    data <- removeMulticollinearFeatures(data)
    
    data <- correctSkewSalePrice(data)
    data <- correctSkewPredictors(data) 

    data <- convertDummyVariables(data)
    data <- removeLessFrequentFeatures(data) 
    
    data
}

bootstrap3 <- function(data){
    #data <- removeOutliers(data)
    
    prices <- data$SalePrice
    tmpData <- data
    tmpData$SalePrice <- NULL
    
    subClasses <- c("20" = "1-Story 1946+",
                    "30" = "1-Story 1945-",
                    "40" = "1-Story attic",
                    "45" = "1,5-Story unfinish",
                    "50" = "1,5-Story finish",
                    "60" = "2-Story 1946+",
                    "70" = "2-Story 1945-",
                    "75" = "2,5-Story",
                    "80" = "Split or multi",
                    "85" = "Split foyer",
                    "90" = "Duplex",
                    "120" = "1-Story PUD 1946+",
                    "150" = "1,5-Story PUD 1946+",
                    "160" = "2-Story PUD 1946+",
                    "180" = "PUD multi",
                    "190" = "2 fam")
    
    tmpData$MSSubClass <- as.factor(tmpData$MSSubClass)
    tmpData$MSSubClass <- revalue(tmpData$MSSubClass, subClasses)
    
    #impute medians for numeric features
    preProc <- preProcess(tmpData, method = "medianImpute")
    tmpData <- predict(preProc, tmpData)
    
    #impute mode for all other remaining NAs
    modes <- tmpData %>% summarise_all(getMode)
    tmpData <- tmpData %>% replace_na(as.list(modes))
    
    tmpData$SalePrice <- prices
    
    #introuduce dummy variables
    preProc <- dummyVars(SalePrice ~ ., data = tmpData)
    tmpData <- as.data.frame(predict(preProc, tmpData))
    
    #remove zero-variance features
    nzv <- nearZeroVar(tmpData)
    tmpData <- tmpData[, -nzv]
    
    tmpData$SalePrice <- prices
    data <- tmpData
    
    data <- correctSkewSalePrice(data)
    data <- correctSkewPredictors(data)
    
    #drop multicollinear features (cor value > 0.8)
    td <- getTrainData(data)
    multicollinear <- cor(data) %>%
        melt() %>%
        filter(Var1 != Var2 & abs(value) > 0.8)
    
    factors <- getFactorData(multicollinear)
    alreadySeen <- list()
    toDrop <- list()
    for(i in seq(length(multicollinear$Var1))){
        var1 <- setdiff(multicollinear$Var1[i], factors)
        var2 <- setdiff(multicollinear$Var2[i], factors)
        
        if(var1 %in% alreadySeen | var2 %in% alreadySeen)
            next
        
        priceCor1 <- cor(td[names(td) == var1], td$SalePrice)
        priceCor2 <- cor(td[names(td) == var2], td$SalePrice)
        
        #drop the feature with least correlation w.r.t. SalePrice
        if(abs(priceCor1) > abs(priceCor2))
            toDrop[[length(toDrop)+1]] <- var2
        else
            toDrop[[length(toDrop)+1]] <- var1
        
        alreadySeen[[length(alreadySeen)+1]] <- var1
        alreadySeen[[length(alreadySeen)+1]] <- var2
    }
    
    data <- data[!(names(data) %in% toDrop)]
    data
}


#Lasso parameters: alpha=1, lambda=0.0015
#GBR parameters: nrounds=300, max_depth=3, eta = 0.1, gamma = c(0.01),colsample_bytree = c(0.75),subsample = c(0.50),min_child_weight = c(0))
#Ridge parameters: alpha = 0, lambda = 0.032
bootstrap4 <- function(data){
    trainData <- getTrainData(data)
    testData <- getTestData(data)
    #-------Removing outliers------
    trainData <- trainData[-c(524, 1299),]  
    data <- rbind(trainData, testData)
    #-------Remove multicollinear------
    data$SalePrice <- log1p(data$SalePrice)
    trainData$SalePrice <- log1p(trainData$SalePrice)
    #-------Correcting skewness of numeric features-------
    numericnames <- names(which(sapply(data, is.numeric)))
    factorsnames <- names(data) %!in% numericnames
    datafactors <- sapply(data[factorsnames], as.factor)
    datanumeric <-data[numericnames]
    data <- cbind(datafactors, datanumeric)
    trainData <- getTrainData(data)
    skewed_feats <- sapply(trainData[numericnames],skew)
    skewed_feats = skewed_feats[skewed_feats > 0.65]
    skewed_feats_names = names(skewed_feats)
    data[skewed_feats_names] = log1p(data[skewed_feats_names])
    #-------Replacing all NAs in factors columns with 'None'-------
    datafactors <- getFactorData(data)
    nafactors<-(colnames(datafactors)[colSums(is.na(datafactors)) > 0])
    data[, nafactors]<- factor(data[, nafactors], levels=c(levels(data[, nafactors]), "None"))
    for (fattore in nafactors) {
      col <- data[fattore]
      col[is.na(col)]<- 'None'
      data[fattore] <- col
    }
    #-------Appending dummies and replacing 'None' with 0,then conversione Character -> Factor-------
    d <- data[, names(datafactors)]
    d[d=='None']=as.integer(0)
    data[, names(datafactors)]<-d
    characters<- which(sapply(data, is.character))
    data[characters]<- as.numeric(unlist(data[characters]))
    data <- appendDummyVariables(data)
    data[names(datafactors)]<- NULL
    #-----------Adding new feature TotalSF-----------
    data$TotalSF=data$GrLivArea+data$X1stFlrSF+data$X2ndFlrSF
    #-----------Replacing all remaining numeric NAs with the mean-----------
    numericnames <- names(which(sapply(data, is.numeric)))
    numericnames <- numericnames[numericnames!="SalePrice"]
    numerici <- data[numericnames]
    for(i in 1:ncol(numerici)){
      numerici[is.na(numerici[,i]), i] <- mean(numerici[,i], na.rm = TRUE)
    }
    data[numericnames] <- numerici
    #----------Drop not relevant features-----------
    notRelevant <- c('X1stFlrSF', 'TotRmsAbvGrd', 'X2ndFlrSF','GarageArea')
    toRemove <- names(data) %in% notRelevant
    data <- numerical[!toRemove]
    #------------Models---------------
    testData <- getTestData(data)
    lasso <- getLassoModel(data)
    ridge <- getRidgeModel(data)
    xgb   <- getGradientBoostingModel(data)
    predictionslasso <- expm1(predict(lasso, testData))
    predictionsridge <- expm1(predict(ridge, testData))
    predictionsxgb   <- expm1(predict(xgb, testData))
    res <- (2*predictionslasso+1.5*predictionsridge+1.5*predictionsxgb)/5
    predictionDF <- data.frame(Id = testIDs, SalePrice = res)
    write.csv(predictionDF, file = "ugo.csv", row.names = FALSE)
}

# here instructions to automatically perform bootstrapping, test on real test set and saving predictions to file

