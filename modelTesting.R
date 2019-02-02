# Authors:  Angelo Di Mambro, Emanuele Giona, Luigi Berducci
# Date:     January 2019
# Purpose:  Library for locally testing models without submitting to the House Prices competition from Kaggle

library(MASS)
library(e1071)

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
        predictions <- expm1(predictions)
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

# Linear model with feature selection using 'backward model selection'
############ DON'T RUN IT BECAUSE IT TAKES A LONG TIME ##############
getLinearModelWithBackwardSelection <- function(data, AREYOUSURE=F){
    train <- getTrainData(data)
    if (AREYOUSURE){
        simpleLM <- lm(SalePrice ~ ., data=train)
        backwardModel <- stepAIC(simpleLM, direction='backward')
    } else{
        backwardModel <- lm(formula = SalePrice ~ LotFrontage + LotArea + Street + Utilities +
        OverallQual + OverallCond + YearBuilt + YearRemodAdd + MasVnrArea +
        BsmtExposure + BsmtFinType1 + BsmtFinSF1 + BsmtFinSF2 + BsmtUnfSF +
        HeatingQC + CentralAir + X2ndFlrSF + GrLivArea + BsmtFullBath +
        FullBath + HalfBath + BedroomAbvGr + KitchenAbvGr + KitchenQual +
        Functional + Fireplaces + GarageType + GarageYrBlt + GarageCars +
        GarageQual + PavedDrive + WoodDeckSF + ScreenPorch + PoolQC +
        MSSubClass20 + MSSubClass30 + MSSubClass50 + MSSubClass60 +
        MSSubClass70 + MSSubClass80 + MSSubClass85 + MSSubClass90 +
        MSSubClass120 + MSSubClass180 + MSZoningFV + MSZoningRH +
        MSZoningRL + MSZoningRM + AlleyPave + LandContourHLS + LandContourLvl +
        LotConfigCulDSac + LotConfigFR2 + LotConfigFR3 + LotConfigInside +
        NeighborhoodCollgCr + NeighborhoodCrawfor + NeighborhoodEdwards +
        NeighborhoodGilbert + NeighborhoodIDOTRR + NeighborhoodMeadowV +
        NeighborhoodMitchel + NeighborhoodNAmes + NeighborhoodNridgHt +
        NeighborhoodNWAmes + NeighborhoodOldTown + NeighborhoodSawyer +
        NeighborhoodSawyerW + NeighborhoodStoneBr + NeighborhoodTimber +
        Condition1Norm + Condition1PosN + Condition1RRAe + Condition1RRAn +
        Condition1RRNn + Condition2Feedr + Condition2Norm + Condition2PosA +
        Condition2PosN + Condition2RRNn + BldgType2fmCon + HouseStyle1.5Unf +
        HouseStyle2.5Unf + RoofStyleGambrel + RoofStyleShed + RoofMatlCompShg +
        RoofMatlMembran + RoofMatlMetal + RoofMatlRoll + `RoofMatlTar&Grv` +
        RoofMatlWdShake + RoofMatlWdShngl + Exterior1stBrkComm +
        Exterior1stBrkFace + Exterior1stHdBoard + Exterior1stPlywood +
        `Exterior1stWd Sdng` + `Exterior2ndBrk Cmn` + Exterior2ndCmentBd +
        Exterior2ndOther + `Exterior2ndWd Sdng` + FoundationPConc +
        FoundationStone + FoundationWood + HeatingGasA + HeatingGasW +
        HeatingWall + ElectricalSBrkr + FenceMnPrv + FenceNone +
        MiscFeatureOthr + MiscFeatureTenC + MoSold5 + MoSold10 +
        SaleTypeConLD + SaleTypeCWD + SaleTypeNew + SaleTypeOth +
        SaleConditionAdjLand + SaleConditionAlloca + SaleConditionNormal,
        data =  train)
    }
    backwardModel
}

getSVM <- function(data){
    set.seed(12345)
    # trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
    trctrl <- trainControl(method = "cv", number = 10)
    train <- getTrainData(data)
    grid <- expand.grid(C=0.88)
    model <- train(SalePrice ~ ., data=train, method="svmLinear", trControl=trctrl, preProces=c("center", "scale"), tuneLength=10, tuneGrid=grid)
}

# --- Lasso ---

getLassoModel <- function(data){
    set.seed(12345)
    train <- getTrainData(data)
    prices <- train$SalePrice
    train$SalePrice <- NULL
    
    control <- trainControl(method="cv", number = 25)
    grid <- expand.grid(alpha = 1, lambda = 0.0015)
    model <- train(x = train, y = prices, method = "glmnet", trControl = control, tuneGrid = grid)
    
    model
}

# --- Ridge ---

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

# --- Elastic Net ---

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

# --- Neural Network ---

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

# --- Extreme Gradient Boosting (XGB) ---

getGradientBoostingModel <- function(data){
    set.seed(12345)
    train <- getTrainData(data)
    prices <- train$SalePrice
    train$SalePrice <- NULL
    
    control <- trainControl(method="cv", number = 10)
    grid <- expand.grid(nrounds = 300, #c(100,200,300),
                        max_depth = 3, #c(3:7),
                        eta = 0.1, #c(0.05, 1),
                        gamma = 0.01,
                        colsample_bytree = 0.75,
                        subsample = 0.50,
                        min_child_weight = 0)
    model <- train(x = train, y = prices, method = "xgbTree", trControl = control, tuneGrid = grid, allowParallel = T)
    
    model
}

# --- Custom ensemble models go here ---

# modelList is a list of model constructors; weights is an array to perform a weighted average of the predictions
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

# produces predictions for the ensemble model
ensemblePredict <- function(ensembleModel, data, checkSkew=T){
    preds <- NULL
    totW <- 0
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
    
    avgPreds <- rowSums(preds) / totW
    avgPreds
}

# Single iteration of CV on an ensemble model.
#
# Parameters:
# ===========
#   -`multiModel`:  list of model constructor
#   -`data`:        complete dataset used
#   -`numIters`:    number of iterations
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

# splits data into k almost-equal sized parts and returns the indexes to perform data splitting as a list
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

# performs k-fold CV with k being the number of base models selected; the column-wise concatenation of each output will then be fed into the specified meta-model
# the variant parameter is a factor with levels:
#   - A: averages the test set predictions to build the meta-test set (default)
#   - B: trains each base model on all folds and the meta-test set is built upon these models' predictions on the test set
# returns a named list containing the following objects:
#   - model = stacked model
#   - predictions = predictions for the test set
# baseModelList is a list of model constructors, and metaModel is a model constructor as well
getStackedRegressor <- function(data, baseModelList, metaModel, variant = "A"){
    n <- length(baseModelList)
    if(n <= 0)
        stop("Base model constructors' list cannot be empty.")
    
    # allowed strategies for producing the actual predictions on the test set, later fed as meta-test set
    variant <- factor(variant, levels = c("A", "B"))
    if(is.na(variant))
        stop("Unknown variant value. Accepted variants are: A, B")
    
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
            baseModel <- baseModelList[[j]](currTrain)
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

    # build the meta-model and train it on its meta-training set
    model <- metaModel(metaTrain)

    # performs predictions on the meta-test set
    predictions <- predictSalePrices(model, metaTest)

    # named list containing both the stacked model and the actual predictions
    ret <- list("model" = model, "predictions" = predictions)

    ret
}

# Old methods
old_predictSalePrices <- function(model, data, checkSkew = T){
    test <- getTestData(data)
    test$SalePrice <- NULL
    predictions <- predict(model, test)
    if (checkSkew == T && SKEWCORRECTION==TRUE)
        predictions <- exp(predictions)
    predictions
}


