# Loading all the required packages
library(ggplot2)
library(data.table)
library(dplyr)
library(caret)
library(tm)
library(pls)
library(AppliedPredictiveModeling)
library(e1071) 
library(corrplot)
library(lattice)
library(nnet)
library(earth)
library(kernlab)
library(pROC)
library(MASS)

#------------------------------------------
# Preprocessing
#------------------------------------------

# Reading data
hdTrain   <- read.csv('train.csv')
hdTest   <- read.csv('test.csv')
hdDesc    <- read.csv("product_descriptions.csv")
hdAttr <- read.csv('attributes.csv')

# Building a text preprocessing function
TextPreProcess<- function(text){
      a<-Corpus(VectorSource(text))
      a<-tm_map(a, content_transformer(tolower))
      a<-tm_map(a, PlainTextDocument)
      a<-tm_map(a, removePunctuation)
      a<-tm_map(a, removeWords,stopwords("english"))
      a<-tm_map(a, stemDocument)
      sapply(1:length(text), function(y) as.character(a[[y]][[1]]))
}

# Applying Text Preprocessing function on the dataset
hdDesc$NewDesc    <- TextPreProcess(hdDesc$product_description)
hdTrain$NewTitle  <- TextPreProcess(hdTrain$product_title)
hdTrain$NewSearch <- TextPreProcess(hdTrain$search_term)
hdTest$NewTitle  <- TextPreProcess(hdTest$product_title)
hdTest$NewSearch <- TextPreProcess(hdTest$search_term)

# Joining Train and Description file
hdTrain<- left_join(hdTrain, hdDesc)
hdTest<- left_join(hdTest, hdDesc)


#----------------------------------------------------
# Creating numerical predictors from text predictors
#----------------------------------------------------

# Creating a function to generate numerical predictors from text predictors 
searchTermMatch <- function(searchTerm,title,desc){
      noTitle <- 0
      noDesc <- 0
      searchTerm <- unlist(strsplit(searchTerm," "))
      noWords <- length(searchTerm)
      for(i in 1:length(searchTerm)){
            pattern <- paste("(^| )",searchTerm[i],"($| )",sep="")
            noTitle <- noTitle + grepl(pattern,title,perl=TRUE,ignore.case=TRUE)
            noDesc <- noDesc + grepl(pattern,desc,perl=TRUE,ignore.case=TRUE)
            percentTitMatch  <- noTitle/ noWords
            percentDescMatch <- noDesc/  noWords
      }
      return(c(noWords,noTitle,noDesc, percentTitMatch, percentDescMatch))
}

# Applying predictor conversion function on the dataset
train_words <- as.data.frame(t(mapply(searchTermMatch, hdTrain$NewSearch, hdTrain$NewTitle, hdTrain$NewDesc)))
hdTrain$nmatch_title <- train_words[,2]
hdTrain$nwords <- train_words[,1]
hdTrain$nmatch_desc <- train_words[,3]
hdTrain$percentTitMatch <- train_words[,4]
hdTrain$percentDescMatch <- train_words[,5]

# Replacing all NA, NaN, Inf values in the dataset
hdTrain <- data.table(hdTrain)
invisible(lapply(names(hdTrain),function(.name) set(hdTrain, which(is.infinite(hdTrain[[.name]])), j = .name,value =0)))
hdTrain[is.na(hdTrain)] <- 0

# Applying predictor conversion function on the test dataset
test_words <- as.data.frame(t(mapply(searchTermMatch, hdTest$NewSearch, hdTest$NewTitle, hdTest$NewDesc)))
hdTest$nmatch_title <- test_words[,2]
hdTest$nwords <- test_words[,1]
hdTest$nmatch_desc <- test_words[,3]
hdTest$percentTitMatch <- test_words[,4]
hdTest$percentDescMatch <- test_words[,5]

# Replacing all NA, NaN, Inf values in the dataset
hdTest <- data.table(hdTest)
invisible(lapply(names(hdTest),function(.name) set(hdTest, which(is.infinite(hdTest[[.name]])), j = .name,value =0)))
hdTest[is.na(hdTest)] <- 0


#************************************************************************************************************
# Linear Regression
#************************************************************************************************************

# Defining Control for Cross Validation
fitControl<- trainControl(method="repeatedCV",
                          number=10,
                          repeats=10,
                          allowParallel = T,
                          predictionBounds = c(1,3))

#-------------------------------
# Gradient Boosting (gbm)
#-------------------------------
gbmModel <- train(relevance~nmatch_title+nmatch_desc+nwords+percentTitMatch+percentDescMatch,data=hdTrain,
                  method="gbm", trControl = fitControl, verbose=TRUE)
gbmModel

test_relevance <- predict(gbmModel,hdTest)
test_relevance <- ifelse(test_relevance>3,3,test_relevance)
test_relevance <- ifelse(test_relevance<1,1,test_relevance)

submission <- data.frame(id=hdTest$id,relevance=test_relevance)
write.csv(submission,"Ashok_GBM_Submission.csv")
print(Sys.time()-t)