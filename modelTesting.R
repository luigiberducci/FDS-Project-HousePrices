# Authors:  Angelo Di Mambro, Emanuele Giona, Luigi Berducci
# Date:     January 2019
# Purpose:  Library for locally testing models without submitting to the House Prices competition from Kaggle

# --- Data utils ---

# Given a complete dataset, returns a scaled version of it
#
# Parameters:
# ===========
#   -`data`:    complete dataset (train + test sets)
#
# Return:
# =======
#   -`data`: scaled data into the [0,1] range
scaleData <- function(data){
    maxVals <- apply(data, 2, max, na.rm = T)
    minVals <- apply(data, 2, min, na.rm = T)
    data <- as.data.frame(scale(data, center = minVals, scale = maxVals - minVals))
    data
}

# Given a complete dataset, returns the training set only
#
# Parameters:
# ===========
#   -`data`:    complete dataset (train + test sets)
#   -`scaled`:  true if values have to be scaled in the [0,1] range
#
# Return:
# =======
#   -`train`: returns the portion of the dataset with has actual values for SalePrice
getTrainData <- function(data, scaled = F){
    if(scaled == T)
        data <- scaleData(data)
    
    train <- data[!is.na(data$SalePrice), ]
    train
}

# Given a complete dataset, returns the test set only
#
# Parameters:
# ===========
#   -`data`:    complete dataset (train + test sets)
#   -`scaled`:  true if values have to be scaled in the [0,1] range
#
# Return:
# =======
#   -`test`: returns the portion of the dataset with has NA values for SalePrice
getTestData <- function(data, scaled = F){
    if(scaled == T)
        data <- scaleData(data)
    
    test <- data[is.na(data$SalePrice), ]
    test
}

# Returns K partitions in train/test set of data according to the balance value.
#
# Parameters:
# ==========
#   -`data`:    dataset from which produce partitions
#   -`k`:       number of partitions
#   -`balance`: factor of balancing in training and test set
#
# Return:
# =======
#   -`partitions`: list of partitions train/test set
getKDataPartitions <- function(data, k, balance=0.8){
    partitions <- list()
    for(i in 1:k){
        # Split in train/test data
        trainSamples <- data$SalePrice %>% createDataPartition(p=balance, list=FALSE)
        trainData <- data[trainSamples, ]
        testData  <- data[-trainSamples, ]
        
        partitions[[i]] <- list(trainData = trainData, testData = testData)
    }
    partitions
}

# Splits data in K folds, to be used in leave-one-out training
#
# Parameters:
# ===========
#   -`data`:    dataset to be split
#   -`k`:       number of parts
#
# Return:
# =======
#   -`split`: list containing k elements, each specifying start and end indices to perform the splitting
getKFolds <- function(data, k){
    n <- nrow(data)
    chunkSize <- as.integer(n/k) + 1
    
    split <- list()
    for(i in 1:k){
        start <- (i-1) * chunkSize
        end <- i * chunkSize - 1
        if(end > n)
            end <- n
        
        split[[i]] <- c(start, end)
    }
    
    split
}

# --- Prediction functions ---

# Performs predictions of test set SalePrice values
#
# Parameters:
# ===========
#   -`model`:       model to use for predictions (compatible with caret train)
#   -`data`:        complete dataset (train + test sets)
#   -`checkSkew`:   true if SalePrice values have been processed with log1p
#
# Return:
# =======
#   -`predictions`: result of caret predict function
predictSalePrices <- function(model, data, checkSkew = T){
    test <- getTestData(data)
    test$SalePrice <- NULL
    predictions <- predict(model, test)
    if (checkSkew == T && SKEWCORRECTION==TRUE)
        predictions <- expm1(predictions)
    predictions
}

# Returns predictions performed by a custom ensemble model
#
# Parameters:
# ===========
#   -`ensembleModel`:   trained custom ensemble model
#   -`data`:            complete dataset (train + test sets)
#   -`checkSkew`:       true if SalePrice values have been processed with log1p
#
# Return:
# =======
#   -`avgPreds`: weighted average of each base model's predictions
ensemblePredict <- function(ensembleModel, data, checkSkew=T){
    preds <- NULL
    totW <- 0
    epsilon <- 0.000000001    # To avoid Div by 0
    
    for(i in 1:length(ensembleModel$models)){
        p <- predictSalePrices(ensembleModel$models[[i]], data, checkSkew)
        w <- ensembleModel$weights[[i]]
        p <- w * p
        totW <- totW + w
        
        if(is.null(preds))
            preds <- p
        else
            preds <- cbind(preds, p)
    }
    
    avgPreds <- rowSums(preds) / (totW+epsilon)
    avgPreds
}

# --- Simple models ---

# Returns a simple linear model using R's lm function
#
# Parameters:
# ===========
#   -`data`:        complete dataset (train + test sets)
#   -`isMetaModel`: true if this model has to be used as a meta-model in a stacked regressor
#
# Return:
# =======
#   -`model`: a trained model
getSimpleLinearModel <- function(data, isMetaModel=F){
    set.seed(12345)
    train <- getTrainData(data)
    model <- lm(SalePrice ~ ., data=train)
    
    model
}

# Returns a model based on Lasso regression
#
# Parameters:
# ===========
#   -`data`:        complete dataset (train + test sets)
#   -`isMetaModel`: true if this model has to be used as a meta-model in a stacked regressor
#
# Return:
# =======
#   -`model`: a caret trained model
getLassoModel <- function(data, isMetaModel=F){
    set.seed(12345)
    train <- getTrainData(data)
    prices <- train$SalePrice
    train$SalePrice <- NULL
    
    control <- trainControl(method="cv", number = 25)
    if(isMetaModel)
        grid <- expand.grid(alpha = 1, lambda = 0.002)
    else
        grid <- expand.grid(alpha = 1, lambda = 0.0015)
    model <- train(x = train, y = prices, method = "glmnet", trControl = control, tuneGrid = grid)
    
    model
}

# Returns a model based on Ridge regression
#
# Parameters:
# ===========
#   -`data`:        complete dataset (train + test sets)
#   -`isMetaModel`: true if this model has to be used as a meta-model in a stacked regressor
#
# Return:
# =======
#   -`model`: a caret trained model
getRidgeModel <- function(data, isMetaModel=F){
    set.seed(12345)
    train <- getTrainData(data)
    prices <- train$SalePrice
    train$SalePrice <- NULL
    
    control <- trainControl(method="cv", number = 25)
    if(isMetaModel)
        grid <- expand.grid(alpha = 0, lambda = 0.0375)
    else
        grid <- expand.grid(alpha = 0, lambda = 0.032)
    model <- train(x = train, y = prices, method = "glmnet", trControl = control, tuneGrid = grid)
    
    model
}

# Returns a model based on Elastic Net regression
#
# Parameters:
# ===========
#   -`data`:        complete dataset (train + test sets)
#   -`isMetaModel`: true if this model has to be used as a meta-model in a stacked regressor
#
# Return:
# =======
#   -`model`: a caret trained model
getENModel <- function(data, isMetaModel=F){
    set.seed(12345)
    train <- getTrainData(data)
    prices <- train$SalePrice
    train$SalePrice <- NULL
    
    control <- trainControl(method="cv", number = 25)
    if(isMetaModel)
        grid <- expand.grid(alpha = 0.5, lambda = 0.0035)
    else
        grid <- expand.grid(alpha = 0.5, lambda = 0.0015)
    model <- train(x = train, y = prices, method = "glmnet", trControl = control, tuneGrid = grid)
    
    model
}

# Returns a model based on Extreme Gradient Boosting
#
# Parameters:
# ===========
#   -`data`:        complete dataset (train + test sets)
#   -`isMetaModel`: true if this model has to be used as a meta-model in a stacked regressor
#
# Return:
# =======
#   -`model`: a caret trained model
getXGBModel <- function(data, isMetaModel=F){
    set.seed(12345)
    train <- getTrainData(data)
    prices <- train$SalePrice
    train$SalePrice <- NULL
    
    control <- trainControl(method="cv", number = 10)
    if(isMetaModel)
        grid <- expand.grid(nrounds = 200,
                            max_depth = 3,
                            eta = 0.05,
                            gamma = 0.01,
                            colsample_bytree = 0.75,
                            subsample = 0.50,
                            min_child_weight = 0)
    else
        grid <- expand.grid(nrounds = 300,
                            max_depth = 3,
                            eta = 0.1,
                            gamma = 0.01,
                            colsample_bytree = 0.75,
                            subsample = 0.50,
                            min_child_weight = 0)
    
    model <- train(x = train, y = prices, method = "xgbTree", trControl = control, tuneGrid = grid, allowParallel = T)
    
    model
}

# Returns a model based on Support Vector Machines
#
# Parameters:
# ===========
#   -`data`:        complete dataset (train + test sets)
#   -`isMetaModel`: true if this model has to be used as a meta-model in a stacked regressor
#
# Return:
# =======
#   -`model`: a caret trained model
getSVM <- function(data, isMetaModel=F){
    set.seed(12345)
    # trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
    trctrl <- trainControl(method = "cv", number = 10)
    train <- getTrainData(data)
    if(isMetaModel)
        grid <- expand.grid(C=0.88)
    else
        grid <- expand.grid(C=0.88) #DEFAULT TUNING
    model <- suppressWarnings(train(SalePrice ~ ., data=train, method="svmLinear", trControl=trctrl, preProces=c("center", "scale"), tuneLength=10, tuneGrid=grid))
    
    model
}

# --- Custom models ---

# Returns a custom ensemble model based on a weighted averaging technique
#
# Parameters:
# ===========
#   -`modelList`:   list of model constructors to use as base models
#   -`weights`:     list of weights to apply to each base model's predictions during the weighted average
#   -`data`:        complete dataset (train + test sets)
#
# Return:
# =======
#   -`avgModel`: a custom ensemble model, containing each caret trained base models and the associated weights
createEnsembleModel <- function(modelList, weights, data){
    n <- length(modelList)
    
    if(n <= 0)
        stop("Model constructors' list cannot be empty.")
    
    if(n != length(weights))
        stop("Model constructors' list and weights do not have the same lenght.")
    
    models <- list()
    for(i in 1:length(modelList))
        models[[i]] <- modelList[[i]](data)
    
    avgModel <- list(type = "Regression")
    avgModel$models <- models
    avgModel$weights <- weights
    
    avgModel
}

# Returns a custom model based on stacked regression
#
# Parameters:
# ===========
#   -`data`:            complete dataset (train + test sets)
#   -`baseModelList`:   list of model constructors to use as base models (trained with leave-one-out technique)
#   -`metaModel`:       model constructor to use as meta models
#   -`variant`:         factor; levels: A (average partially-trained base models' predictions to feed the meta-model), B (feed fully-trained base models' predictions to the meta-model)
#   -`verbose`:         true if debug information has to be printed out
#
# Return:
# =======
#   -`ret`: a custom model, containing a caret trained meta-model and the actual predictions on the test set
getStackedRegressor <- function(data, baseModelList, metaModel, variant = "A", verbose = F){
    n <- length(baseModelList)
    if(n <= 0)
        stop("Base model constructors' list cannot be empty.")
    
    # allowed strategies for producing the actual predictions on the test set, later fed as meta-test set
    variant <- factor(variant, levels = c("A", "B"))
    if(is.na(variant))
        stop("Unknown variant value. Accepted variants are: A, B")
    
    #Debug
    if(verbose)
        print("[Debug] Start stacked regressor")
    
    # split training data into n folds
    train <- getTrainData(data)
    trainFolds <- getKFolds(train, n)
    
    # actual test set to perform base model's predictions on
    test <- getTestData(data)
    
    # holds base model's predictions during training, to be used for the meta-model's training
    baseTrainPredictions <- list()
    
    # holds base model's predictions for the actual test set
    # variant A case: for each base model, stores a list of predictions (one for each model trained on the training set without 1 heldout set)
    # variant B case: for each base model, stores predictions of a model trained on the whole training set
    baseTestPredictions <- list()
    
    # perform leave-one-out training of base models
    for(i in 1:n){ # selects the heldout set
        #Debug
        if(verbose)
            print("[Debug] Iteration on K Fold")
        
        # current training set is the whole training set except the heldout set
        currTrain <- NULL
        for(j in 1:n){
            if(j == i)
                next()
            
            fold = trainFolds[[j]]
            start = fold[1]
            end = fold[2]
            if(is.null(currTrain))
                currTrain <- train[start:end,]
            else
                currTrain <- rbind(currTrain, train[start:end,])
        }
        
        # current test set is the heldout set
        fold <- trainFolds[[i]]
        start = fold[1]
        end = fold[2]
        currTest <- train[start:end,]
        currTest$SalePrice <- NULL
        
        # train the base models, store predictions for the heldout set in order to produce the meta-training set
        # store base models' predictions on the test set to later use them as meta-test set
        for(j in 1:n){ # selects the base model
            # Debug
            if(verbose)
                print("[Debug] Iteration on base model")
            # baseModel <- suppressWarnings(baseModelList[[j]](currTrain))
            baseModel <- baseModelList[[j]](currTrain)
            
            if(verbose)
                print("[Debug] predict current fold with base model")
            trainPredictions <- predict(baseModel, currTest)
            
            testPredictions <- NULL
            switch(as.character(variant),
                   "A" = {
                       # variant A stores each base model's predictions on the test set, for each k-fold training routine
                       testPredictions <- predictSalePrices(baseModel, test, checkSkew = F)
                   },
                   "B" = {
                       #variant B does nothing here
                   })
            
            if(j > length(baseTrainPredictions)){
                baseTrainPredictions[[j]] <- trainPredictions
                
                switch(as.character(variant),
                       "A" = {
                           # variant A needs to initialize the inner list (the one holding each k-fold predictions)
                           baseTestPredictions[[j]] <- list(testPredictions)
                       },
                       "B" = {
                           #variant B does nothing here
                       })
            }
            else{
                baseTrainPredictions[[j]] <- c(baseTrainPredictions[[j]], trainPredictions)
                
                switch(as.character(variant),
                       "A" = {
                           # variant A simply adds the new predictions in tail of the existing list
                           baseTestPredictions[[j]][[i]] <- testPredictions
                       },
                       "B" = {
                           #variant B does nothing here
                       })
            }
        }
    }
    
    #Debug 
    if(verbose)
        print("[Debug] Collect base models' predictions")
    
    switch(as.character(variant),
           "A" = {
               # variant A averages each base model's predictions in order to get that models' entry in the meta-test set
               averagedPredictions <- list()
               for(i in 1:n){ # selects the base model
                   predsList <- baseTestPredictions[[i]]
                   predsNum <- length(predsList)
                   predictions <- 0
                   for(j in 1:predsNum)
                       predictions <- predictions + predsList[[j]]
                   predictions <- predictions / predsNum
                   
                   # store the averaged predictions as this model's predictions in the meta-test set
                   if(i > length(averagedPredictions))
                       averagedPredictions[[i]] <- predictions
                   else
                       averagedPredictions[[i]] <- c(averagedPredictions[[i]], predictions)
               }
               
               # uniformity with variant B's case
               baseTestPredictions <- averagedPredictions
           },
           "B" = {
               #variant B trains each base model on the full training set and uses these new predictions as their entries in the meta-test set
               for(i in 1:n){ # selects the base model
                   # baseModel <- suppressWarnings(baseModelList[[i]](train))
                   baseModel <- baseModelList[[i]](train)
                   predictions <- predictSalePrices(baseModel, test, checkSkew = F)
                   
                   if(i > length(baseTestPredictions))
                       baseTestPredictions[[i]] <- predictions
                   else
                       baseTestPredictions[[i]] <- c(baseTestPredictions[[i]], predictions)
               }
           })
    
    # holds columns' names for the conversion to dataframe
    columns <- NULL
    
    # holds predictions of base models to be used as a meta-training set
    metaTrain <- NULL
    
    # holds predictions of base models to be used as a meta-testing set
    metaTest <- NULL
    
    # column-wise merging each base model's predictions on both training and test set
    for(i in 1:n){
        if(is.null(metaTrain)){
            metaTrain <- baseTrainPredictions[[i]]
            metaTest <- baseTestPredictions[[i]]
        }
        else{
            metaTrain <- cbind(metaTrain, baseTrainPredictions[[i]])
            metaTest <- cbind(metaTest, baseTestPredictions[[i]])
        }
        
        modelName <- paste("Model", i, sep = "")
        if(is.null(columns))
            columns <- modelName
        else
            columns <- c(columns, modelName)
    }
    
    # build the meta-training set as a dataframe
    columns <- c(columns, "SalePrice")
    metaTrain <- cbind(metaTrain, as.list(train$SalePrice))
    metaTrain <- as.data.frame(metaTrain)
    colnames(metaTrain) <- columns
    metaTrain <- as.data.frame(sapply(metaTrain, as.numeric))
    
    # build the meta-test set as a dataframe
    metaTest <- cbind(metaTest, as.list(test$SalePrice))
    metaTest <- as.data.frame(metaTest)
    colnames(metaTest) <- columns
    metaTest <- as.data.frame(sapply(metaTest, as.numeric))
    
    #Debug 
    if(verbose)
        print("[Debug] Train the metamodel")
    
    # build the meta-model and train it on its meta-training set
    model <- metaModel(metaTrain, isMetaModel=T)
    
    # performs predictions on the meta-test set
    predictions <- predictSalePrices(model, metaTest)
    
    # named list containing both the stacked model and the actual predictions
    ret <- list("model" = model, "predictions" = predictions)
    
    ret
}

# --- Cross-validation and other testing ---

# Performs a single step of cross-validation and returns performance stats
#
# Parameters:
# ===========
#   -`model`:           model to test
#   -`data`:            test set
#   -`stackedModel`:    true if the model to test is a stacked model
#   -`recipe`:          recipe to pass down the stacked model's constructor
#   -`completeData`:    complete dataset (train + test sets)
#
# Return:
# =======
#   -`res`: dataframe containing performance stats
crossValidation <- function(model, data, stackedModel = F, recipe = NULL, completeData = NULL){
    # Save the groundtruth to future comparison
    groundTruth <- data$SalePrice
    if(stackedModel == F)
        data$SalePrice <- NA
    else if(is.null(completeData))
        stop("Stacked regressor requires the complete dataset, even for CV.")
    
    pred  <- NULL
    if(stackedModel == T){
        if(is.null(recipe) || is.null(recipe$baseModelList) || !is.list(recipe$baseModelList) || is.null(recipe$metaModel))
            stop("Unable to perform CV on a stacked model without a valid recipe.")
        
        models <- recipe$baseModelList
        metaModel <- recipe$metaModel
        var <- NULL
        if(!is.null(recipe$variant))
            var <- recipe$variant
        
        stack <- getStackedRegressor(completeData,
                                     baseModelList = models,
                                     metaModel = metaModel,
                                     variant = var)
        pred <- stack$predictions
        pred <- log1p(pred)
    }
    else
        pred <- predictSalePrices(model, data, checkSkew = F)
    
    res   <- data.frame( R2 = R2(pred, groundTruth),
                         RMSE = RMSE(pred, groundTruth),
                         MAE = MAE(pred, groundTruth))
    res
}

# Performs cross-validation and returns performance stats
#
# Parameters:
# ===========
#   -`modelConstructor`:    model constructor of the model to test
#   -`data`:                complete dataset (train + test sets)
#   -`nTimes`:              iterations of CV to perform
#   -`stackedModel`:        true if the model to test is a stacked model
#   -`recipe`:              recipe to pass down the stacked model's constructor
#
# Return:
# =======
#   -`finalRes`: dataframe containing performance stats per CV iteration
iterateCV <- function(modelConstructor, data, nTimes, stackedModel = F, recipe = NULL){
    
    # Work only on train data
    allTrain <- getTrainData(data, scaled = F)
    
    #create partitions for cross validation (avoiding the same splitting due to set.seed)
    sets <- getKDataPartitions(allTrain, nTimes)
    
    finalRes <- data.frame()
    
    #for each partion, train a new model on the selected portion of training data and test against the other portion
    for(i in 1:length(sets)){
        set <- sets[[i]]
        model <- NULL
        cvData <- NULL
        if(stackedModel == F)
            model <- modelConstructor(set$trainData)
        else{
            cvTest <- set$testData
            cvTest$SalePrice <- NA
            cvData <- rbind(set$trainData, cvTest)
        }
        
        currentRes <- crossValidation(model, set$testData, stackedModel = stackedModel, recipe = recipe, completeData = cvData)
        finalRes <- rbind(finalRes, currentRes)
    }
    
    finalRes
}

# Given a set of partitions, train an ensemble model for each partition.
#
# Parameters:
# ===========
#   -`models`:      list of base-model constructor which compose the ensemble
#   -`weights`:     list of weights for each base-model
#   -`partitions`:  list of partitions train/test set
#
# Return:
# =======
#   -`ensembleList`: list of ensemble models trained on the associated partitions
trainEnsembleModelOnPartitions <- function(models, weights, partitions){
    ensembleList <- list()
    for(i in 1:length(partitions)){
        currentPartition <- partitions[[i]]
        trainData <- currentPartition$trainData
        
        ensemble <- createEnsembleModel(models, weights, trainData) 
        ensembleList[[i]] <- ensemble
    }
    ensembleList
}

# Given a set of partitions and a list ensemble models, return the result (R2, RMSE, MAE) of the predictions.
#
# Parameters:
# ===========
#   -`ensembleList`:    list of trained ensemble models
#   -`partitions`:      list of partitions train/test set
# 
# Return:
# =======
#   -`finalRes`: dataframe with a row for each test and with statistics as columns
testEnsembleModelOnPartitions <- function(ensembleList, partitions){
    finalRes <- data.frame()
    for(i in 1:length(partitions)){
        currentPartition <- partitions[[i]]
        testData  <- currentPartition$testData
        groundTruth <- testData$SalePrice
        testData$SalePrice <- NA
        ensemble <- ensembleList[[i]]
        
        pred <- ensemblePredict(ensemble, testData, checkSkew=F)
        
        res   <- data.frame( R2 = R2(pred, groundTruth),
                             RMSE = RMSE(pred, groundTruth),
                             MAE = MAE(pred, groundTruth))
        
        finalRes <- rbind(finalRes, res)
    }
    finalRes
}

# Single iteration of CV on an ensemble model.
#
# Parameters:
# ===========
#   -`multiModel`:      list of model constructor
#   -`modelWeights`:    list of model weights
#   -`data`:            complete dataset used
#   -`numIters`:        number of iterations
iterateCrossValidationEnsembleModel <- function(multiModel, modelWeights, data, numIters){
    data <- getTrainData(data)

    print("Create partitioning...")
    partitions <- getKDataPartitions(data, numIters)

    print("Train an ensemble for each partition...")
    ensembleList <- trainEnsembleModelOnPartitions(multiModel, modelWeights, partitions)

    print("Run predictions...")
    result <- testEnsembleModelOnPartitions(ensembleList, partitions)
    result
}

# find optimal weights for Ensemble Model
testEnsembleWeights <- function(data){
    print("Starting")
    print(Sys.time())
    
    models <- list(getSimpleLinearModel, getLassoModel, getRidgeModel, getENModel, getXGBModel, getSVM)
    weights <- c(1, 1, 1, 1, 1, 1)
    
    allWeights <- c(0, 0.25, 0.5, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5, 4, 4.5, 5)
    permutation <- permutations(length(allWeights), length(models), allWeights, repeats.allowed=T)
    numPermutations <- length(permutation[,1])
    numIterations <- 10
    
    cat(sprintf("[Info] Ensemble of %d models. Test %d weight permutations\n", length(models), length(permutation)))
    
    data <- getFinalFeatures(data)
    data <- getTrainData(data)
    
    partitions <- getKDataPartitions(data, numIterations)
    finalRes <- data.frame()
    for(i in 1:numIterations){
        cat(sprintf("[Info] Iter=%d\n", i))
        partition <- partitions[[i]]
        
        ensemble <- createEnsembleModel(models, weights, partition$trainData)
        groundTruth <- partition$testData$SalePrice
        partition$testData$SalePrice <- NA
        minRMSE <- 1
        for(j in 1:numPermutations){
            if(j%%1000==0)
                cat(sprintf("\t[Info] Perm=%d\n", j))
            
            weights <- permutation[j,]
            ensemble$weights <- weights
            
            pred <- ensemblePredict(ensemble, partition$testData, checkSkew=F)
            rmse <- RMSE(pred, groundTruth)
            if(rmse<minRMSE){
                minWeights <- weights
                minRMSE <- rmse
            }
        }
        ensemble$weights <- minWeights
        bestPred <- ensemblePredict(ensemble, partition$testData, checkSkew=F)
        res   <- data.frame(w1 = minWeights[1],
                            w2 = minWeights[2],
                            w3 = minWeights[3],
                            w4 = minWeights[4],
                            R2 = R2(bestPred, groundTruth),
                            RMSE = RMSE(bestPred, groundTruth),
                            MAE = MAE(bestPred, groundTruth))
        finalRes <- rbind(finalRes, res)
        str(finalRes)
    }
    
    print("Ending")
    print(Sys.time())
    
    finalRes
}

# check for optimal Ensemble Model
testOptimalWeightsMinimum <- function(data){
    data <- getFinalFeatures(data)
    
    models <- c(getLassoModel, getRidgeModel, getXGBModel, getSVM)
    weights <- c(0.5, 0.5, 3.5, 5)
    
    ensemble <- createEnsembleModel(models, weights, data)
    pred <- ensemblePredict(ensemble, data)
    savePredictionsOnFile(testIDs, pred, "out/2019_02_02_4models_min_optimal_weights.csv")
    pred
}

printf <- function(...) cat(sprintf(...))

printStackedTestInformation <- function(baseModels, metaModel, nIters){
    printf("[Info] Testing Stacked Regression with %d base models\n", length(baseModels))
    printf("[Info] Test: %d iteration of Cross-validation\n\n", nIters)
}

# compare variant A vs variant B performances of the same Stacked Regressor, using CV
testStackedRegressor<- function(data, metaModelConstructor, baseModelList, nIters=10){
    # Debug
    printStackedTestInformation(baseModelList, metaModelConstructor, nIters)
    
    data <- getFinalFeatures(data)
    
    baseModels <- baseModelList
    metaModel <- metaModelConstructor
    variants <- c("A", "B")
    nIterations <- nIters
    
    resForBothVariants <- data.frame()
    for (var in variants){
        printf("[Info] Testing variant %s (%s)\n", var, Sys.time())
        stackedRecipe <- list( baseModelList = baseModels, 
                               metaModel = metaModel, 
                               variant = var)
        res <- iterateCV(getStackedRegressor, data, nIterations, TRUE, stackedRecipe)
        res$Variant <- var
        resForBothVariants <- rbind(resForBothVariants, res)
    }
    
    # Debug
    print(resForBothVariants)
    resForBothVariants
}

# pretty print stats of a Stacked Regressor, grouped by variant
printStatisticsResultDF <- function(df){
    min  <- df %>% group_by(Variant) %>% summarise_all(min)  %>% as.data.frame
    mean <- df %>% group_by(Variant) %>% summarise_all(mean) %>% as.data.frame
    max  <- df %>% group_by(Variant) %>% summarise_all(max)  %>% as.data.frame
    
    min$Stat  <- "Min"
    mean$Stat <- "Mean"
    max$Stat  <- "Max"
    
    all <- rbind(min, mean, max)
    res <- data.frame(Variant=all$Variant, Stat=all$Stat, R2=all$R2, RMSE=all$RMSE, MAE=all$MAE)
    res[order(res$Variant),]
}
