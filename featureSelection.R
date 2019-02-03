# WHOLE FILE DEPRECATED ?

source("modelTesting.R")

getRelevantByCaretModel <- function(modelConstructor, data){
    model <- modelConstructor(data)
    imp <- varImp(model)
    imp$names <- row.names(imp$importance)
    impNames <- imp$names[imp$importance>0]
    data[, c(impNames, "SalePrice")]
}

removeNearZeroVarFeatures <- function(data){
    nzv <- nearZeroVar(data)
    data[, -nzv]
}
