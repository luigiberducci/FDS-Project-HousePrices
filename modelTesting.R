# Authors:  Angelo Di Mambro, Emanuele Giona, Luigi Berducci
# Date:     January 2019
# Purpose:  Library for locally testing models without submitting to the House Prices competition from Kaggle

iterateCrossValidationNTimes <- function(data, nTimes){
  finalRes <- data.frame()
  for(i in 1:nTimes){
    currentRes <- crossValidation(data)
    finalRes <- rbind(finalRes, currentRes)
  }
  finalRes
}

crossValidation <- function(data){
  # Work only on train data
  data <- getOnlyRelevantFeatures(data)
  allTrain <- getTrainData(data) 
  
  # Split in train/test data
  trainSamples <- allTrain$SalePrice %>% createDataPartition(p=0.8, list=FALSE)
  trainData <- allTrain[trainSamples, ]
  testData  <- allTrain[-trainSamples, ]
  
  # Save the groundtruth to future comparison
  groundTruth <- testData$SalePrice
  testData$SalePrice <- NA
  
  # Build the model and predict prices
  model <- getSimpleLinearModel(trainData)
  pred  <- predictSalePrices(model, testData)
  res   <- data.frame( R2 = R2(pred, groundTruth),
                       RMSE = RMSE(pred, groundTruth),
                       MAE = MAE(pred, groundTruth))
  res
}

savePredictionsOnFile <- function(ids, pred, outputPath){
  predictionDF <- data.frame(Id = ids, SalePrice = pred)
  write.csv(predictionDF, file = outputPath, row.names = FALSE)
}

predictSalePrices <- function(model, data){
  test <- getTestData(data)
  test$SalePrice <- NULL
  predictions <- predict(model, test)
  if (skewCorrection==TRUE)
    predictions <- exp(predictions)
  predictions
}

getSimpleLinearModel <- function(data){
  train <- getTrainData(data)
  model <- lm(SalePrice ~ ., data=train)
  model
}
