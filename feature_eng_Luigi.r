#Libraries
library(plyr)
library(psych)

#Dataset
train = read.csv("data/train.csv", stringsAsFactor=FALSE)
test  = read.csv("data/test.csv", stringsAsFactor=FALSE)

#Feature blocks
outside = c("PavedDrive", "Fence", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "ScreenPorch",  "PoolArea", "PoolQC")
garage = c("GarageType", "GarageYrBlt", "GarageFinish", "GarageCars", "GarageArea", "GarageQual", "GarageCond")
rooms = c("BedroomAbvGr", "TotRmsAbvGrd")
kitchen = c("KitchenAbvGr", "KitchenQual")
bathrooms = c("BsmtFullBath", "BsmtHalfBath", "FullBath", "HalfBath")
general = c("OverallQual", "OverallCond", "Functional")
all = c(general, bathrooms, kitchen, rooms, garage, outside, 'SalePrice')

#Function
getFeatures <- function(data, featList){
    newData <- data[featList]
    return(newData)
}

getNumericalData <- function(data){
    newData <- data[, getNumericalFields(data)]
    return(newData)
}

getNumericalFields <- function(data) {
    newData <- which(sapply(data, is.numeric))
    return(newData)
}

getFactor <- function(data) {
    newData <- which(sapply(data, is.factor))
    return(newData)
}

getNAReport <- function(data) {
    missingFeatures <- which(colSums(is.na(data)) > 0)
    return(missingFeatures)
}

convertToOrdinal <- function(data){
    data <- convertFunctional(data)
    data <- convertKitchenQual(data)
    data <- convertBathRooms(data)
    data <- convertGarageFinish(data)
    data <- convertGarageQual(data)
    data <- convertGarageCond(data)
    data <- convertGarageCars(data)
    data <- convertGarageArea(data)
    data <- convertFence(data)
    data <- convertPoolQC(data)
    return(data)
}

convertToFactors <- function(data){
    data <- convertGarageType(data)
    data <- convertGarageYrBlt(data)
    data <- convertPavedDrive(data)
    return(data)
}

convertFunctional <- function(data) {
    data$Functional[is.na(data$Functional)] <- 'Typ'
    replace <- c('Typ'=7, 'Min1'=6, 'Min2'=5, 'Mod'=4, 'Maj1'=3, 'Maj2'=2, 'Sev'=1, 'Sal'=0)
    data$Functional <- as.integer(revalue(data$Functional, replace, warn_missing=FALSE))
    return(data)
}

convertKitchenQual <- function(data) {
    data$KitchenQual[is.na(data$KitchenQual)] <- 'TA'
    replace <- c('Ex'=4, 'Gd'=3, 'TA'=2, 'Fa'=1, 'Po'=0)
    data$KitchenQual <- as.integer(revalue(data$KitchenQual, replace, warn_missing=FALSE))
    return(data)
}

convertBathRooms <- function(data) {
    data$FullBath[is.na(data$FullBath)] <- 0
    data$HalfBath[is.na(data$HalfBath)] <- 0
    data$BsmtFullBath[is.na(data$BsmtFullBath)] <- 0
    data$BsmtHalfBath[is.na(data$BsmtHalfBath)] <- 0
    return(data)
}

convertGarageFinish <- function(data) {
    data$GarageFinish[is.na(data$GarageFinish)] <- 'Miss'
    replace <- c('Fin'=3, 'RFn'=2, 'Unf'=1, 'Miss'=0)
    data$GarageFinish <- as.integer(revalue(data$GarageFinish, replace, warn_missing=FALSE))
    return(data)
}

convertGarageYrBlt <- function(data) {
    # med <- median(train$GarageYrBlt[!is.na(train$GarageYrBlt)])
    data$GarageYrBlt[is.na(data$GarageYrBlt)] <- 0
    data$GarageYrBlt <- as.factor(data$GarageYrBlt)
    return(data)
}

convertGarageQual <- function(data) {
    data$GarageQual[is.na(data$GarageQual)] <- 'Miss'
    replace <- c('Ex'=5, 'Gd'=4, 'TA'=3, 'Fa'=2, 'Po'=1, 'Miss'=0)
    data$GarageQual <- as.integer(revalue(data$GarageQual, replace, warn_missing=FALSE))
    return(data)
}

convertGarageCond <- function(data) {
    data$GarageCond[is.na(data$GarageCond)] <- 'Miss'
    replace <- c('Ex'=5, 'Gd'=4, 'TA'=3, 'Fa'=2, 'Po'=1, 'Miss'=0)
    data$GarageCond <- as.integer(revalue(data$GarageCond, replace, warn_missing=FALSE))
    return(data)
}

convertFence <- function(data) {
    data$Fence[is.na(data$Fence)] <- 'Miss'
    replace <- c('GdPrv'=4, 'MnPrv'=3, 'GdWo'=2, 'MnWw'=1, 'Miss'=0)
    data$Fence <- as.integer(revalue(data$Fence, replace, warn_missing=FALSE))
    return(data)
}

convertPoolQC <- function(data) {
    data$PoolQC[is.na(data$PoolQC)] <- 'Miss'
    replace <- c('Ex'=4, 'Gd'=3, 'TA'=2, 'Fa'=1, 'Miss'=0)
    data$PoolQC <- as.integer(revalue(data$PoolQC, replace, warn_missing=FALSE))
    return(data)
}

convertGarageCars <- function(data) {
    data$GarageCars[ is.na(data$GarageCars) ] <- 0
    return(data)
}

convertGarageArea <- function(data) {
    data$GarageArea[ is.na(data$GarageArea) ] <- 0
    return(data)
}

convertGarageType <- function(data) {
    data$GarageType[ is.na(data$GarageType) ] <- 'None'
    data$GarageType <- as.factor(data$GarageType)
    return(data)
}

convertPavedDrive <- function(data) {
    data$PavedDrive[ is.na(data$PavedDrive) ] <- 'None'
    data$PavedDrive <- as.factor(data$PavedDrive)
    return(data)
}

checkGarageConsistency <- function(data){
    garageVars = c("GarageType", "GarageYrBlt", "GarageFinish", "GarageCars", "GarageArea", "GarageQual",    "GarageCond")
    
    print(data[!is.na(data$GarageType) & (is.na(data$GarageType) | is.na(data$GarageYrBlt) | is.na(data$GarageFinish) | is.na(data$GarageCars) | data$GarageCars==0 | data$GarageArea==0 | is.na(data$GarageArea) | is.na(data$GarageQual) | is.na(data$GarageCond)), garageVars])
    print(data[!is.na(data$GarageYrBlt) & (is.na(data$GarageType) | is.na(data$GarageYrBlt) | is.na(data$GarageFinish) | is.na(data$GarageCars) | data$GarageCars==0 | data$GarageArea==0 | is.na(data$GarageArea) | is.na(data$GarageQual) | is.na(data$GarageCond)), garageVars])
    print(data[!is.na(data$GarageFinish) & (is.na(data$GarageType) | is.na(data$GarageYrBlt) | is.na(data$GarageFinish) | is.na(data$GarageCars) | data$GarageCars==0 | data$GarageArea==0 | is.na(data$GarageArea) | is.na(data$GarageQual) | is.na(data$GarageCond)), garageVars])
    print(data[data$GarageCars>0 & (is.na(data$GarageType) | is.na(data$GarageYrBlt) | is.na(data$GarageFinish) | is.na(data$GarageCars) | data$GarageCars==0 | data$GarageArea==0 | is.na(data$GarageArea) | is.na(data$GarageQual) | is.na(data$GarageCond)), garageVars])
    print(data[data$GarageArea>0 & (is.na(data$GarageType) | is.na(data$GarageYrBlt) | is.na(data$GarageFinish) | is.na(data$GarageCars) | data$GarageCars==0 | data$GarageArea==0 | is.na(data$GarageArea) | is.na(data$GarageQual) | is.na(data$GarageCond)), garageVars])
    print(data[!is.na(data$GarageQual) & (is.na(data$GarageType) | is.na(data$GarageYrBlt) | is.na(data$GarageFinish) | is.na(data$GarageCars) | data$GarageCars==0 | data$GarageArea==0 | is.na(data$GarageArea) | is.na(data$GarageQual) | is.na(data$GarageCond)), garageVars])
    print(data[!is.na(data$GarageCond) & (is.na(data$GarageType) | is.na(data$GarageYrBlt) | is.na(data$GarageFinish) | is.na(data$GarageCars) | data$GarageCars==0 | data$GarageArea==0 | is.na(data$GarageArea) | is.na(data$GarageQual) | is.na(data$GarageCond)), garageVars])
}

checkPoolConsistency <- function(data){
    poolVars <- c("PoolArea", "PoolQC")

    print( data[data$PoolArea>0 & is.na(data$PoolQC), poolVars] )
    print( data[data$PoolArea==0 & !is.na(data$PoolQC), poolVars] )
}

missingToMedian <- function(data){
    fillWithMedian <- function(x) replace(x, is.na(x), median(x, na.rm = TRUE))
    data <- sapply(data, function(x){
            if(is.numeric(x)){
                fillWithMedian(x)
            } else {
                x
            }
        }
    )
    return(as.data.frame(data))
}

convertAllCHRToFactor <- function(data){
    character_vars <- lapply(data, class) == "character"
    data[, character_vars] <- lapply(data[, character_vars], as.factor)
    return(data)
}

addFeatureBathrooms <- function(data) {
    prices <- data$SalePrice
    data$SalePrice <- NULL
    data$TotBathRms <- data$BsmtFullBath + 0.5*data$BsmtHalfBath + data$FullBath + 0.5*data$HalfBath
    data$SalePrice <- prices
    return(data)
}

featureSelection <- function(data) {
    selected<- c("OverallQual", "FullBath", "KitchenAbvGr", "GarageCars", "TotRmsAbvGrd", "TotBathRms")
    data$OverallCond <- NULL
    data$Functional <- NULL
    data$BsmtFullBath <- NULL
    data$BsmtHalfBath <- NULL
    data$HalfBath <- NULL
    data$BedroomAbvGr <- NULL
    data$GarageType <- NULL
    data$GarageYrBlt <- NULL
    data$GarageFinish <- NULL
    data$GarageArea <- NULL
    data$GarageQual <- NULL
    data$GarageCond <- NULL
    data$PavedDrive <- NULL
    data$Fence <- NULL
    data$WoodDeckSF <- NULL
    data$OpenPorchSF <- NULL
    data$EnclosedPorch <- NULL
    data$ScreenPorch <- NULL
    data$PoolArea <- NULL
    data$PoolQC <- NULL
    return(data)
}

runCompleteProcess <- function(data){
    data <- convertToOrdinal(data)
    data <- convertToFactors(data)
    data <- addFeatureBathrooms(data)
#    data <- featureSelection(data)
#    data <- getNumericalData(data)
}

writeCSV <- function(test, predictions, outputFile){
    labels <- test$Id
    df <- data.frame(Id = labels, SalePrice = predictions)
    write.csv(df, file = outputFile, row.names = F)
}
