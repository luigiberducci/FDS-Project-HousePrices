# Angelo
# Group 5: 6
#YearBuilt: Original construction date
#YearRemodAdd: Remodel date (same as construction date if no remodeling or additions)
#-MoSold: Month Sold (MM)
#-YrSold: Year Sold (YYYY)
#-SaleType: Type of sale
#-SaleCondition: Condition of sale

# Group 6: 9
#-RoofStyle: Type of roof
#-RoofMatl: Roof material
#-Exterior1st: Exterior covering on house
#-Exterior2nd: Exterior covering on house (if more than one material)
#-MasVnrType: Masonry veneer type
#-MasVnrArea: Masonry veneer area in square feet
#-ExterQual: Evaluates the quality of the material on the exterior 
#-ExterCond: Evaluates the present condition of the material on the exterior
#-Foundation: Type of foundation

# Group 7: 11
#-BsmtQual: Evaluates the height of the basement
#-BsmtCond: Evaluates the general condition of the basement
#-BsmtExposure: Refers to walkout or garden level walls
#-BsmtFinType1: Rating of basement finished area
#-BsmtFinSF1: Type 1 finished square feet
#-BsmtFinType2: Rating of basement finished area (if multiple types)
#-BsmtFinSF2: Type 2 finished square feet
#-BsmtUnfSF: Unfinished square feet of basement area
#-TotalBsmtSF: Total square feet of basement area
#LowQualFinSF: Low quality finished square feet (all floors)
#GrLivArea: Above grade (ground) living area square feetlibrary(knitr)

#----Libraries----
library(ggplot2)
library(plyr)
library(dplyr)
library(corrplot)
library(caret)
library(gridExtra)
library(scales)
library(Rmisc)
library(ggrepel)
library(psych)
library(xgboost)

#----Datasets----
train <- read.csv("C:\\Users\\Angelo\\Desktop\\FDS_Project\\train.csv", stringsAsFactors = F)
test <- read.csv("C:\\Users\\Angelo\\Desktop\\FDS_Project\\test.csv", stringsAsFactors = F)

#----Levels Vectors----
Qualities <- c('None'=0, 'Po' = 1, 'Fa' = 2, 'TA' = 3, 'Gd' = 4, 'Ex' = 5)
Exposure <- c('None'=0, 'No'=1, 'Mn'=2, 'Av'=3, 'Gd'=4)
FinType <- c('None'=0, 'Unf'=1, 'LwQ'=2, 'Rec'=3, 'BLQ'=4, 'ALQ'=5, 'GLQ'=6)
Masonry <- c('None'=0, 'BrkCmn'=0, 'BrkFace'=1, 'Stone'=2)

#----My features----
myfeatures <- c("YearBuilt","YearRemodAdd","MoSold","YrSold","SaleType","SaleCondition","RoofStyle","RoofMatl","Exterior1st","Exterior2nd","MasVnrType","MasVnrArea","ExterQual","ExterCond","Foundation","BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinSF1","BsmtFinType2","BsmtFinSF2","BsmtUnfSF","TotalBsmtSF","LowQualFinSF","GrLivArea","SalePrice")
train$Id <- NULL
test$Id <- NULL
test$SalePrice <- NA
mydata <- rbind(train, test)


#returns the mode of a collection of values, ignoring NAs
getMode <- function(values){
  uniques <- unique(values)
  uniques <- uniques[!is.na(uniques)]
  maxFreqID <- which.max(tabulate(match(values,uniques)))
  uniques[maxFreqID]}
  
  
#returns values as factors, filling NAs with the mode or with the specified argument
getFactors <- function(values, replaceNA = "NA"){
  if(!is.factor(values))
  {
    values[is.na(values)] <- ifelse(replaceNA == "NA", getMode(values), replaceNA)
    values <- as.factor(values)
  }
  values
}
#returns values as integers, enconding them via the given dictionary, filling NAs with the mode or with the specified argument
getIntegers <- function(values, dictionary, replaceNA = "NA"){
  if(!is.integer(values))
  {
    values[is.na(values)] <- ifelse(replaceNA == "NA", getMode(values), replaceNA)
    values <- as.integer(dictionary[values])
  }
  values
}


mydata$MoSold <- getFactors(mydata$MoSold)
mydata$SaleType <- getFactors(mydata$SaleType)
mydata$SaleCondition <- getFactors(mydata$SaleCondition)
mydata$RoofStyle <- getFactors(mydata$RoofStyle)
mydata$RoofMatl <- getFactors(mydata$RoofMatl)
mydata$Exterior1st <- getFactors(mydata$Exterior1st)
mydata$Exterior2nd <- getFactors(mydata$Exterior2nd)
mydata$Foundation <- getFactors(mydata$Foundation)
mydata$ExterQual <- getIntegers(mydata$ExterQual,Qualities)
mydata$ExterCond <- getIntegers(mydata$ExterCond,Qualities)
mydata$MasVnrType <- getIntegers(mydata$MasVnrType,Masonry, "None")
mydata$BsmtQual <- getIntegers(mydata$BsmtQual,Qualities, "None")
mydata$BsmtCond <- getIntegers(mydata$BsmtCond,Qualities, "None")
mydata$BsmtExposure <- getIntegers(mydata$BsmtExposure,Exposure, "None")
mydata$BsmtFinType1 <- getIntegers(mydata$BsmtFinType1,FinType, "None")
mydata$BsmtFinType2 <- getIntegers(mydata$BsmtFinType2,FinType, "None")
mydata$BsmtFinSF1[is.na(mydata$BsmtFinSF1)] <-0
mydata$BsmtFinSF2[is.na(mydata$BsmtFinSF2)] <-0
mydata$BsmtUnfSF[is.na(mydata$BsmtUnfSF)] <-0
mydata$TotalBsmtSF[is.na(mydata$TotalBsmtSF)] <-0
mydata$MasVnrArea[is.na(mydata$MasVnrArea)] <-0

