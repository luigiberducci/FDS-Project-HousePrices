#Libraries
library(plyr)
library(psych)

#Dataset
train = read.csv("data/train.csv", stringsAsFactor=FALSE)
test  = read.csv("data/test.csv", stringsAsFactor=FALSE)

#Feature blocks
outside = c("PavedDrive", "Fence", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "ScreenPorch",  "PoolArea", "PoolQC")
garage = c("GarageType", "GarageYrBlt", "GarageFinish", "GarageCars", "GarageArea", "GarageQual", "GarageCond")
rooms = c("BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd")
kitchen = c("KitchenQual")
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
    data <- convertGarageFinish(data)
    data <- convertGarageQual(data)
    data <- convertGarageCond(data)
    data <- convertFence(data)
    data <- convertPoolQC(data)
    return(data)
}

convertToFactors <- function(data){
    data <- convertGarageType(data)
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

convertGarageFinish <- function(data) {
    data$GarageFinish[is.na(data$GarageFinish)] <- 'Miss'
    replace <- c('Fin'=3, 'RFn'=2, 'Unf'=1, 'Miss'=0)
    data$GarageFinish <- as.integer(revalue(data$GarageFinish, replace, warn_missing=FALSE))
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
