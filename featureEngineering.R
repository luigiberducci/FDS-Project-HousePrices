# Authors:  Angelo Di Mambro, Emanuele Giona, Luigi Berducci
# Date:     January 2019
# Purpose:  Library for feature engineering the datasets for the House Prices competition from Kaggle

# --- Global flags ---
SKEWCORRECTION    <<- FALSE
NEWFEATBATH       <<- FALSE
NEWFEATCARSXAREA  <<- FALSE
NEWFEATRECENTG    <<- FALSE
NEWFEATRECENTTYPE <<- FALSE
NEWFEATTOTALSF    <<- FALSE
DUMMIESVAR <<- c()

# --- Helper functions ---
`%!in%` = Negate(`%in%`)

getCharFields <- function(data) {
    newData <- which(sapply(data, is.character))
    names(newData)
}

getFactorData <- function(data) {
    newData <- data[, getFactorFields(data)]
    newData
}

getNumericalData <- function(data) {
    newData <- data[, getNumericalFields(data)]
    newData
}

getNotNumericalData <- function(data) {
    newData <- data[, getNotNumericalFields(data)]
    newData
}

getFactorFields <- function(data) {
    newData <- which(sapply(data, is.factor))
    names(newData)
}

getNumericalFields <- function(data) {
    newData <- which(sapply(data, is.numeric))
    names(newData)
}

getNotNumericalFields <- function(data){
    newData <- data[, names(data) %!in% getNumericalFields(data)]
    names(newData)
}

# --- Data tidying ---

removeOutliers <- function(data){
    trainData <- getTrainData(data)
    testData <- getTestData(data)
    
    trainData <- trainData[-c(524, 1299),]
    
    data <- rbind(trainData, testData)
    data
}

handleNA <- function(data){
    data <- replaceNAwtNone(data)
    data <- replaceNoneWtZero(data)
    data <- encodeCharAsNumeric(data)
}

replaceNAwtNone <- function(data){
    factData <- getNotNumericalData(data)
    
    factWtNA <- colnames(factData)[colSums(is.na(factData)) > 0]
    data[, factWtNA] <- factor(data[, factWtNA], levels=c(levels(data[, factWtNA]), "None"))
    for (fact in factWtNA) {
        col <- data[fact]
        col[is.na(col)]<- 'None'
        data[fact] <- col
    }
    data
}

replaceNoneWtZero <- function(data){
    notNumData <- getNotNumericalData(data)
    d <- data[, names(notNumData)]
    d[d=='None']=as.integer(0)
    data[, names(notNumData)] <- d
    characters<- which(sapply(data, is.character))
    data[characters]<- as.numeric(unlist(data[characters]))
    data
}

encodeCharAsNumeric <- function(data){
    characters  <- getCharFields(data)
    data[characters] <- as.numeric(unlist(data[characters]))
    data
}

encodeCharAsFactors <- function(data){
    numData  <- getNumericalData(data)
    
    numNames <- getNumericalFields(data)
    notNumNames <- getNotNumericalFields(data)
    
    factData <- sapply(data[notNumNames], as.factor)
    data <- cbind(factData, numData)
}

# --- Feature engineering ---

correctSkewnessSalePrice <- function(data){
    SKEWCORRECTION <<- TRUE
    data$SalePrice <- log1p(data$SalePrice)
    data
}

correctSkewnessPredictors <- function(data){
    # skew correction based only on training data
    trainData <- getTrainData(data)
    numNames <- getNumericalFields(data)

    skewness <- sapply(trainData[numNames], skew)
    skewedFeatureNames = names(skewness[skewness > 0.65])

    data[skewedFeatureNames] = log1p(data[skewedFeatureNames])
    data
}

handleSkewness <- function(data){
    data <- correctSkewnessSalePrice(data)
    data <- encodeCharAsFactors(data)
    data <- correctSkewnessPredictors(data)
}

appendDummyVariables <- function(data){
    prices <- data$SalePrice
    data$SalePrice <- NULL
    fact <- getFactorData(data)
    dummies <- as.data.frame(model.matrix(~.-1, fact))
    DUMMIESVAR <<- names(dummies)
    
    data <- cbind(data, dummies)
    data$SalePrice <- prices
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
    
    data$RecentGarage[is.na(data$GarageYrBlt)] <- 0
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

addFeatureTotalSF <- function(data) {
    prices <- data$SalePrice
    data$SalePrice <- NULL
    data$TotalSF <- data$GrLivArea + data$X1stFlrSF + data$X2ndFlrSF
    data$SalePrice <- prices
    data
}

addNewFeatures <- function(data, totBathRms=F, carsXarea=F, recentGarage=F, totalSF=F){
    if(totBathRms){
        data <- addFeatureBathrooms(data)
        NEWFEATBATH <<- TRUE
    }
    if(carsXarea){
        data <- addFeatureCarsXArea(data)
        NEWFEATCARSXAREA <<- TRUE
    }
    if(recentGarage){
        data <- addFeatureRecentGarage(data)
        NEWFEATRECENTG <<- TRUE
    }
    if(totalSF){
        data <- addFeatureTotalSF(data)
        NEWFEATTOTALSF <<- TRUE
    }
    
    data
}

replaceRemainingNAwtMean <- function(data){ 
    numericnames <- names(which(sapply(data, is.numeric)))
    numericnames <- numericnames[numericnames!="SalePrice"]
    numerics <- data[numericnames]
    for(i in 1:ncol(numerics)){
        numerics[is.na(numerics[,i]), i] <- mean(numerics[,i], na.rm = TRUE)
    }
    data[numericnames] <- numerics
    data
}

# --- Feature selection ---

removeMulticollinearFeatures <- function(data){
    multicollinear <- c("X1stFlrSF", "X2ndFlrSF", "GarageArea", "TotRmsAbvGrd", "GarageCars", "FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath")
    data <- data[!colnames(data) %in% multicollinear]
    data
}

# iteratively drop least important features, minimizing the given model's average RMSE
importanceSelection <- function(data, modelConstructor, maxRounds = 10, verbose = F){
    round <- 0
    
    model <- modelConstructor(data)
    rmse <- mean(model$results$RMSE)
    
    while(round < maxRounds){
        if(verbose)
            print(paste("RMSE (", round, "): ", rmse))
        
        imp <- varImp(model)$importance
        features <- rownames(imp)
        zeroImpPositions <- which(imp$Overall == 0)
        zeroImpFeatures <- features[zeroImpPositions]
        
        newData <- data[!(names(data) %in% zeroImpFeatures)]
        newModel <- modelConstructor(newData)
        newRmse <- mean(newModel$results$RMSE)
        
        if(newRmse < rmse){
            data <- newData
            model <- newModel
            rmse <- newRmse
        }
        else{
            if(verbose)
                print(paste("RMSE did not improve. Stopping with RMSE: ", rmse))
            break
        }
        if(verbose)
            print("")
        
        round <- round + 1
    }
    
    data
}
