################################################################################################                     
                     # Ashok Vardhan Kari
                     # OR 568 - Spring 2016
                     # Home Depot Product Search Term Relevance (Final Project)
################################################################################################

# Installing all the required packages
install.packages("ggplot2")
install.packages("data.table")
install.packages("dplyr")
install.packages("caret")
install.packages("tm")
install.packages("pls")
install.packages("AppliedPredictiveModeling")
install.packages("e1071")
install.packages("corrplot")
install.packages("lattice")
install.packages("nnet")
install.packages("earth")
install.packages("kernlab")
install.packages("pROC")

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
      
      # Joining Train and Description file
      hdTrain<- left_join(hdTrain, hdDesc)

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
# Simple Linear Regression (lm)
#-------------------------------
      lmModel1<- train(relevance~nmatch_title+nmatch_desc+nwords, data=hdTrain, method="lm", 
                      trControl=fitControl, verbose=TRUE)
      lmModel1
      
      #   RMSE       Rsquared   RMSE SD      Rsquared SD
      # 0.5032457  0.1118881  0.003922922  0.005915459
      
      # Linear regression with percentages            
      lmModel2<- train(relevance~nmatch_title+nmatch_desc+nwords+percentTitMatch+percentDescMatch, 
             data=hdTrain, method="lm", trControl=fitControl, verbose=TRUE)
      lmModel2
      
      #   RMSE       Rsquared   RMSE SD      Rsquared SD
      # 0.4985872  0.1283324  0.003780619  0.008842439
      
      
#-------------------------------
# Generalized Linear Regression (glm)
#-------------------------------
      glmModel<- train(relevance~nmatch_title+nmatch_desc+nwords+percentTitMatch+percentDescMatch, 
                       data=hdTrain, method="glm", trControl=fitControl)
      glmModel
      
      #  RMSE       Rsquared   RMSE SD      Rsquared SD
      # 0.4984968  0.1282527  0.004881802  0.01166843 
      

#-------------------------------
# Gradient Boosting (gbm)
#-------------------------------
      trees <- 500
      gbmModel <- train(relevance~nmatch_title+nmatch_desc+nwords+percentTitMatch+percentDescMatch,data=hdTrain,
                        method="gbm", trControl = fitControl, verbose=TRUE)
      gbmModel
      # RMSE        Rsquared 
      # 0.5113095   0.1390016

      
#-------------------------------
# Robust Linear Regression (rlm)
#-------------------------------
      rlmModel<- train(relevance~nmatch_title+nmatch_desc+nwords+percentTitMatch+percentDescMatch, 
                       data=hdTrain, method="rlm", trControl=fitControl, verbose=TRUE)
      rlmModel
      # rlmImp <- varImp(rlmModel, scale = FALSE)
      # RMSE       Rsquared  RMSE SD     Rsquared SD
      # 0.4994083  0.128146  0.00470332  0.007401202
      
         
#-------------------------------
# Partial Least Squares (pls)
#-------------------------------
      set.seed(100)
      plsModel <- train(relevance~nmatch_title+nmatch_desc+nwords+percentTitMatch+percentDescMatch,
                        data=hdTrain, method = "pls",trControl = fitControl)
      plsModel
      # ncomp  RMSE       Rsquared    RMSE SD      Rsquared SD
      # 1      0.5139186  0.07394995  0.003354892  0.005901616
      # 2      0.5015259  0.11796460  0.003520219  0.006308798
      # 3      0.5001331  0.12285665  0.003615514  0.006737014
      
      
      
#-------------------------------
# Principle Component Regression (pcr)
#-------------------------------
      set.seed(100)
      pcrModel <- train(relevance~nmatch_title+nmatch_desc+nwords+percentTitMatch+percentDescMatch,
                       data=hdTrain, method = "pcr",  trControl = fitControl)
      pcrModel
      # ncomp  RMSE       Rsquared    RMSE SD      Rsquared SD
      # 1      0.5309136  0.01161179  0.002875307  0.002455526
      # 2      0.5018954  0.11665042  0.003076449  0.006194712
      # 3      0.5017411  0.11719877  0.003092877  0.006253774
      
      # Plotting the RMSE and the models
      plsResamples <- plsModel$results
      plsResamples$Model <- "PLS"
      pcrResamples <- pcrModel$results
      pcrResamples$Model <- "PCR"
      plsPlotData <- rbind(plsResamples, pcrResamples)
      
      xyplot(RMSE ~ ncomp,
             data = plsPlotData,
             #aspect = 1,
             xlab = "# Components",
             ylab = "RMSE (Cross-Validation)",       
             col = c("blue","red"),
             auto.key = list(columns = 2),
             groups = Model,
             type = c("o", "g"))
    
        
#-------------------------------
# Penalized Model (LASSO Regression)
#-------------------------------      
      enetGrid <- expand.grid(lambda = c(0, 0.01, .1), fraction = seq(.05, 1, length = 20))
      set.seed(100)
      enetModel <- train(relevance~nmatch_title+nmatch_desc+nwords+percentTitMatch+percentDescMatch,
                         data=hdTrain, method = "enet", tuneGrid = enetGrid,
                        trControl = fitControl, preProc = c("center", "scale"))
      enetModel
      plot(enetModel)
      # lambda  fraction     RMSE       Rsquared   RMSE SD      Rsquared SD
      #  0.00    1.00      0.4985888  0.1282516  0.003010145  0.006822792
      
  
          
#-------------------------------      
# Finding and Plotting Importance of Variables
#-------------------------------  
      lmImp1 <- varImp(lmModel1, scale = FALSE)
      lmImp2 <- varImp(lmModel2, scale = FALSE)
      glmImp <- varImp(glmModel, scale = FALSE)
      gbmImp <- varImp(gbmModel, scale = FALSE)
      #rlmImp <- varImp(rlmModel, scale = FALSE)
      plsImp <- varImp(plsModel, scale = FALSE)
      pcrImp <- varImp(pcrModel, scale = FALSE)
      enetImp <- varImp(enetModel, scale = FALSE)
      
      par(mfrow = c(2,2))
      layout(matrix(1:2, nrow = 2))
      plot(lmImp1, scales = list(y = list(cex = .95)), main = "Linear Regression (3 Predictors)")
      plot(lmImp2, scales = list(y = list(cex = .95)), main = "Linear Regression (5 Predictors)")
      plot(glmImp, scales = list(y = list(cex = .95)), main = "Generalized Linear Regression")
      plot(gbmImp, scales = list(y = list(cex = .95)), main = "Gradient Boosting")
      plot(rlmImp, scales = list(y = list(cex = .95)), main = "Robust Linear Regression")
      plot(plsImp, scales = list(y = list(cex = .95)), main = "Partial Least Squares")
      plot(pcrImp, scales = list(y = list(cex = .95)), main = "Principle Component Regression")
      plot(enetImp, scales = list(y = list(cex = .95)), main = "Penalized Model (LASSO Regression)")
          
#************************************************************************************************************
                                    # Non Linear Regression
#************************************************************************************************************

      #----------------------------------------
      # K-Nearest Neighbors (Not Working as of now)
      #----------------------------------------
#       set.seed(100)
#       # Without specifying train control, the default is bootstrap 
#       knnModel = train(relevance~nmatch_title+nmatch_desc+nwords+percentTitMatch+percentDescMatch,
#                        data=hdTrain, method="knn",
#                        preProc=c("center","scale"),
#                        tuneLength=10)
#       
#       knnModel
#       
#       # plot the RMSE performance against the k
#       plot(knnModel$results$k, knnModel$results$RMSE, type="o",xlab="# neighbors",ylab="RMSE", main="KNNs for Friedman Benchmark")
#       
#----------------------------------------
# Neural Networks (Not working as of now)
#----------------------------------------
#       
      nnGrid = expand.grid( .decay=c(0,0.01,0.1), .size=1:5 ) # decay is the penalty and size is no.of hidden models.
      
      set.seed(100)
      nnetModel = train(relevance~nmatch_title+nmatch_desc+nwords+percentTitMatch+percentDescMatch,
                        data=hdTrain, method="nnet", preProc=c("center", "scale"), 
                        linout=TRUE, trace=FALSE, MaxNWts=10 * (ncol(hdTrain$product_uid)+1) + 10 + 1, 
                        maxit=500, tuneGrid = nnGrid)
      
      nnetModel  
      summary(nnetModel)
#       # Lets see what variables are most important: 
#       varImp(nnetModel)
#       
#----------------------------------------
# Support Vector Machine - SVM Radial (Ran forever)
#----------------------------------------
#       
#       set.seed(100)
#       # tune against the cost C
#       svmRModel = train(relevance~nmatch_title+nmatch_desc+nwords+percentTitMatch+percentDescMatch,
#                         data=hdTrain, method="svmRadial", preProc=c("center", "scale"), tuneLength=20)
#       
#       svmRModel
#       # Lets see what variables are most important: 
#       varImp(svmRModel)
      
#----------------------------------------
# Support Vector Machine - SVM Linear 
#----------------------------------------
      set.seed(100)
      # tune against the cost C
      svmModel = train(relevance~nmatch_title+nmatch_desc+nwords+percentTitMatch+percentDescMatch,
                        data=hdTrain, method="svmLinear", preProc=c("center", "scale"), tuneLength=20)
      
      svmModel
      # RMSE       Rsquared   RMSE SD      Rsquared SD
      # 0.5048327  0.1272532  0.003515008  0.003633967
      
      # Lets see what variables are most important: 
      varImp(svmModel)
     
       
#----------------------------------------
#  MARS model
#----------------------------------------
      
      marsGrid = expand.grid(.degree=1:2, .nprune=2:19)
      set.seed(100)
      marsModel = train(relevance~nmatch_title+nmatch_desc+nwords+percentTitMatch+percentDescMatch,
                        data=hdTrain, method="earth", preProc=c("center", "scale"), tuneGrid=marsGrid)
      
      marsModel
      # degree  nprune  RMSE       Rsquared    RMSE SD      Rsquared SD
      # 2       19      0.4917758  0.15307302  0.001918254  0.003231567
      
      # Lets see what variables are most important: 
      varImp(marsModel)
      
      