# Authors:  Angelo Di Mambro, Emanuele Giona, Luigi Berducci
# Date:     January 2019
# Purpose:  Library for locally testing models without submitting to the House Prices competition from Kaggle

iterateCrossValidationNTimes <- function(modelConstructor, data, nTimes, neuralModel = F){
    #compute max and min SalePrice to scale back NN's predictions
    maxPrice <- 0
    minPrice <- 0
    if(neuralModel == T){
        maxPrice = max(data$SalePrice, na.rm = T)
        minPrice = min(data$SalePrice, na.rm = T)
    }
    
    # Work only on train data
    allTrain <- getTrainData(data, scaled = neuralModel)
    
    #create partitions for cross validation (avoiding the same splitting due to set.seed)
    sets <- list()
    for(i in 1:nTimes){
        # Split in train/test data
        trainSamples <- allTrain$SalePrice %>% createDataPartition(p=0.8, list=FALSE)
        trainData <- allTrain[trainSamples, ]
        testData  <- allTrain[-trainSamples, ]
        
        sets[[i]] <- list(trainData = trainData, testData = testData)
    }
    
    finalRes <- data.frame()
    
    #for each partion, train a new model on the selected portion of training data and test against the other portion
    for(i in 1:length(sets)){
        set <- sets[[i]]
        model <- modelConstructor(set$trainData)
        currentRes <- crossValidation(model, set$testData, neuralModel = neuralModel, maxPrice = maxPrice, minPrice = minPrice)
        finalRes <- rbind(finalRes, currentRes)
    }
    
    finalRes
}

crossValidation <- function(model, data, neuralModel = F, maxPrice = 0, minPrice = 0){
    # Save the groundtruth to future comparison
    groundTruth <- data$SalePrice
    data$SalePrice <- NA

    pred  <- NULL
    if(neuralModel == T)
        pred <- predictNeuralSalePrices(model, data, checkSkew = F, isDataScaled = T, maxPrice = maxPrice, minPrice = minPrice)
    else
        pred <- predictSalePrices(model, data, checkSkew = F)
    
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
    test <- getTestData(data, scaled = !isDataScaled)
    test$SalePrice <- NULL
    
    if(maxPrice == 0)
        maxPrice <- max(data$SalePrice, na.rm = T)
    
    if(minPrice == 0)
        minPrice <- min(data$SalePrice, na.rm = T)

    predictions <- predict(model, test)
    predictions <- predictions * (maxPrice - minPrice) + minPrice
    
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

#Elastic Net model
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
    prices <- train$SalePrice
    train$SalePrice <- NULL
    
    control <- trainControl(method="cv", number = 5)
    grid <- expand.grid(size = 7, #seq(1:10),
                        decay = 0.1) #seq(0.1, 0.5, by = 0.05))
    model <- train(x = train, y = prices, method = "nnet", trControl = control, tuneGrid = grid, trace = F, linout = T)
    
    model
}

#Extreme Gradient Boosting model
getGradientBoostingModel <- function(data){
    set.seed(12345)
    train <- getTrainData(data)
    prices <- train$SalePrice
    train$SalePrice <- NULL
    
    control <- trainControl(method="cv", number = 10)
    grid <- expand.grid(nrounds = 300, #c(100,200,300),
                        max_depth = 6, #c(3:7),
                        eta = 0.05, #c(0.05, 1),
                        gamma = c(0.01),
                        colsample_bytree = c(0.75),
                        subsample = c(0.50),
                        min_child_weight = c(0))
    model <- train(x = train, y = prices, method = "xgbTree", trControl = control, tuneGrid = grid, allowParallel = T)
    
    model
}
