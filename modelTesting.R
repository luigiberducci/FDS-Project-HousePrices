# Authors:  Angelo Di Mambro, Emanuele Giona, Luigi Berducci
# Date:     January 2019
# Purpose:  Library for locally testing models without submitting to the House Prices competition from Kaggle

iterateCrossValidationNTimes <- function(model, data, nTimes){
    finalRes <- data.frame()
    for(i in 1:nTimes){
        currentRes <- crossValidation(model, data)
        finalRes <- rbind(finalRes, currentRes)
    }
    finalRes
}

crossValidation <- function(model, data){
    # Work only on train data
    allTrain <- getTrainData(data) 
    
    # Split in train/test data
    trainSamples <- allTrain$SalePrice %>% createDataPartition(p=0.8, list=FALSE)
    trainData <- allTrain[trainSamples, ]
    testData  <- allTrain[-trainSamples, ]
    
    # Save the groundtruth to future comparison -- 
    groundTruth <- testData$SalePrice
    testData$SalePrice <- NA
    
    # Build the model and predict prices -- now taken in input
    #model <- getSimpleLinearModel(trainData)

    pred  <- predictSalePrices(model, testData, F) #fixes the obscene RMSE/MAE scores given as output (comparison of log SalePrice vs SalePrice)
    res   <- data.frame( R2 = R2(pred, groundTruth),
                         RMSE = RMSE(pred, groundTruth),
                         MAE = MAE(pred, groundTruth))
    res
}

savePredictionsOnFile <- function(ids, pred, outputPath){
    predictionDF <- data.frame(Id = ids, SalePrice = pred)
    write.csv(predictionDF, file = outputPath, row.names = FALSE)
}

predictSalePrices <- function(model, data, checkSkew = T){
    test <- getTestData(data)
    test$SalePrice <- NULL
    predictions <- predict(model, test)
    if (checkSkew == T && SKEWCORRECTION==TRUE)
        predictions <- exp(predictions)
    predictions
}

getTrainData <- function(data){
    train <- data[!is.na(data$SalePrice), ]
    train
}

getTestData <- function(data){
    test <- data[is.na(data$SalePrice), ]
    test
}

# Models

#simple linear model
getSimpleLinearModel <- function(data){
    set.seed(12345)
    train <- getTrainData(data)
    model <- lm(SalePrice ~ ., data=train)
    model
}

getLassoModel <- function(data){
    set.seed(12345)
    train <- getTrainData(data)
    prices <- train$SalePrice
    train$SalePrice <- NULL
    
    control <- trainControl(method="cv", number = 10)
    grid <- expand.grid(alpha = 1, lambda = 0.001)
    model <- train(x = train, y = prices, method = "glmnet", trControl = control, tuneGrid = grid)
    
    model
}

getRidgeModel <- function(data){
    set.seed(12345)
    train <- getTrainData(data)
    prices <- train$SalePrice
    train$SalePrice <- NULL
    
    control <- trainControl(method="cv", number = 10)
    grid <- expand.grid(alpha = 0, lambda = 0.0325)
    model <- train(x = train, y = prices, method = "glmnet", trControl = control, tuneGrid = grid)
    
    model
}

#Elastic Net
getENModel <- function(data){
    set.seed(12345)
    train <- getTrainData(data)
    prices <- train$SalePrice
    train$SalePrice <- NULL
    
    control <- trainControl(method="cv", number = 10)
    grid <- expand.grid(alpha = 0.5, lambda = 0.0015)
    model <- train(x = train, y = prices, method = "glmnet", trControl = control, tuneGrid = grid)
    
    model
}
