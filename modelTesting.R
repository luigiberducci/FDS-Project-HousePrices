# Authors:  Angelo Di Mambro, Emanuele Giona, Luigi Berducci
# Date:     January 2019
# Purpose:  Library for locally testing models without submitting to the House Prices competition from Kaggle

iterateCrossValidationNTimes <- function(model, data, nTimes, neuralModel = F){
    finalRes <- data.frame()
    for(i in 1:nTimes){
        currentRes <- crossValidation(model, data, neuralModel)
        finalRes <- rbind(finalRes, currentRes)
    }
    finalRes
}

crossValidation <- function(model, data, neuralModel = F){
    # Work only on train data
    allTrain <- getTrainData(data, scaled = neuralModel) 
    
    # Split in train/test data
    trainSamples <- allTrain$SalePrice %>% createDataPartition(p=0.8, list=FALSE)
    trainData <- allTrain[trainSamples, ]
    testData  <- allTrain[-trainSamples, ]
    
    # Save the groundtruth to future comparison
    groundTruth <- testData$SalePrice
    testData$SalePrice <- NA

    pred  <- NULL
    if(neuralModel == T){
        maxPrice = max(data$SalePrice, na.rm = T)
        minPrice = min(data$SalePrice, na.rm = T)
        pred <- predictNeuralSalePrices(model, testData, checkSkew = F, isDataScaled = T, maxPrice = maxPrice, minPrice = minPrice)
    }
    else
        pred <- predictSalePrices(model, testData, checkSkew = F)
    
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

#for neural network models only
predictNeuralSalePrices <- function(model, data, checkSkew = T, isDataScaled = F, maxPrice = 0, minPrice = 0){
    test <- getTestData(data, !isDataScaled)
    test$SalePrice <- NULL
    
    if(maxPrice == 0)
        maxPrice <- max(data$SalePrice, na.rm = T)
    
    if(minPrice == 0)
        minPrice <- min(data$SalePrice, na.rm = T)
    
    predictions <- compute(model, test)
    predictions <- predictions$net.result * (maxPrice - minPrice) + minPrice
    
    if (checkSkew == T && SKEWCORRECTION==TRUE)
        predictions <- exp(predictions)
    predictions
}

#scales data in the [0,1] range to help the neural network perform better
scaleData <- function(data){
    maxVals <- apply(data, 2, max, na.rm = T)
    minVals <- apply(data, 2, min, na.rm = T)
    data <- as.data.frame(scale(data, center = minVals, scale = maxVals - minVals))
    data
}

getTrainData <- function(data, scaled = F){
    if(scaled == T)
        data <- scaleData(data)
    
    train <- data[!is.na(data$SalePrice), ]
    train
}

getTestData <- function(data, scaled = F){
    if(scaled == T)
        data <- scaleData(data)
    
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
    
    control <- trainControl(method="cv", number = 25)
    grid <- expand.grid(alpha = 1, lambda = 0.002)
    model <- train(x = train, y = prices, method = "glmnet", trControl = control, tuneGrid = grid)
    
    model
}

getRidgeModel <- function(data){
    set.seed(12345)
    train <- getTrainData(data)
    prices <- train$SalePrice
    train$SalePrice <- NULL
    
    control <- trainControl(method="cv", number = 25)
    grid <- expand.grid(alpha = 0, lambda = 0.032)
    model <- train(x = train, y = prices, method = "glmnet", trControl = control, tuneGrid = grid)

    model
}

#Elastic Net
getENModel <- function(data){
    set.seed(12345)
    train <- getTrainData(data)
    prices <- train$SalePrice
    train$SalePrice <- NULL
    
    control <- trainControl(method="cv", number = 25)
    grid <- expand.grid(alpha = 0.5, lambda = 0.0015)
    model <- train(x = train, y = prices, method = "glmnet", trControl = control, tuneGrid = grid)
    
    model
}

#Neural Net model
getNeuralModel <- function(data){
    set.seed(12345)
    train <- getTrainData(data, scaled = T)
    
    n <- names(train)
    f <- as.formula(paste("SalePrice ~", paste(n[!n %in% "SalePrice"], collapse = " + ")))
    model <- neuralnet(f, data = train, hidden = c(5,3), linear.output = T)

    model
}
