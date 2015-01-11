# Machine learning algorithms on Weight Lifting Exercise Dataset
Dmitry  
11 января 2015 г.  

In given report basic machine learning algorithms are tested on [WLE dataset](http://groupware.les.inf.puc-rio.br/har).

##1. Data processing

I am assuming that you have "pml-training.csv" and "pml-testing.csv" files in your working directory. At first we need to clean the data.


```r
library(e1071)
library(caret)
pmltraining <- read.csv("pml-training.csv", stringsAsFactors=FALSE)
pmltesting <- read.csv("pml-testing.csv", stringsAsFactors=FALSE)

#We do not need them since there is no such rows in pmltesting dataset
pmltraining <- subset(pmltraining, new_window == "no")

#Transform classe and user_name to factors
pmltraining <- transform(pmltraining, classe = as.factor(classe), 
                         user_name = as.factor(user_name))
```

It is a good idea to take time out of "cvtd_timestamp", because there might be a time pattern in performing wrong exercises.


```r
#A function that exctracts time and transforms it in continous variable (in hours)
get_hours <- function(df){
    df$hour <- NULL
    for(i in 1:nrow(df)){
        tmp_h <- as.numeric(substr(df$cvtd_timestamp[i],12,13))
        tmp_m <- as.numeric(substr(df$cvtd_timestamp[i],15,16)) / 60
        df$hour[i] <- round(tmp_h + tmp_m,2)
    }
    return(df)
}

pmltesting <- get_hours(pmltesting)
pmltraining <- get_hours(pmltraining)
```

Now remove all unnecessary variables. "X" is just an id, rawtimestamps are actually timestamp, "new_window" is constant and from cvtdtimestamp we already exctracted hours:


```r
#exclude NA and pointless variables
pointless <- c("X","raw_timestamp_part_1","raw_timestamp_part_2","new_window", "cvtd_timestamp")
mes <- NULL
for (x in colnames(pmltesting)){
    if(is.na(pmltesting[1,x]) == FALSE & !(x %in% pointless)){
        mes <- c(mes,x)
    }
}
rm(x)

pmltesting <- pmltesting[,mes]

#no "problem_id" in training set
pmltraining <- pmltraining[,c("classe",mes[-55])]
```

##2. Model training

Now it is time for training. We will create four different models and compare them with each other. NOTE that i was running this code on 64-bit machine so the results might differ with 32-bit machine, but should be close anyway.


```r
set.seed(13)
inTrain = createDataPartition(pmltraining$classe, p = 0.75)[[1]]
training = pmltraining[ inTrain,]
testing = pmltraining[-inTrain,]

#Since lda and svm are both fast we can easily improve them by default cross validation
fitLDA <- train(data = training, classe ~ ., method = "lda", 
               trControl = trainControl(method = "cv", allowParallel = TRUE))

fitSVM <- svm(classe ~ ., data = training, cross = 10)


#Gradient boosting is much slower so i chose boot632 method for cross validation
fitGBM <- train(data = training, classe ~ ., method = "gbm", verbose = FALSE, 
                trControl = trainControl(method = "boot632", allowParallel = TRUE))

#And now most hardcore method. Adding cross validation would be an overkill, even this #training session run for about 8 hours
fitRF <- train(data = training, classe ~ ., method = "rf", 
               prox = TRUE)

#Make prediction on test data set
pLDA <- predict(fitLDA,testing)
pSVM <- predict(fitSVM,testing)
pGBM <- predict(fitGBM,testing)
pRF <- predict(fitRF,testing)

#Get accuracy
LDAacc <- confusionMatrix(pLDA,testing$classe)$overall[1]
SVMacc <- confusionMatrix(pSVM,testing$classe)$overall[1]
GBMacc <- confusionMatrix(pGBM,testing$classe)$overall[1]
RFacc <- confusionMatrix(pRF,testing$classe)$overall[1]
```



Check results:


```r
c(LDAacc,SVMacc,GBMacc,RFacc)
```

```
## [1] 0.8490212 0.9518950 0.9908372 0.9987505
```

Random forest model is a clear winner with unbelievable 0.999 accuracy (missed ***6 out of 4802*** predictions). But is comes with a price, this is model comparsion in terms of resources:

- LDA: few seconds to train, 5.4 Mb;

- SVM: about a minute, 6.4 Mb;

- GBM: about 40 min, 22.6 Mb;

- RF: about **8 hours**, **1.6 Gb**.

I also tried combining predictors using (60:20:20) split and training combined classifier on validation set, but in vain: there was a drastic drop (about 10%) in accuracy on testing set.

##3. Results

Of course I used my RF model to make a submission. It got 20/20 correct answers, but what is funny - even LDA model would get 20/20. LDA, GBM and RF got the same results, only SVM got 19/20.
