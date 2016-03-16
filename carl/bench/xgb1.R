#https://www.kaggle.com/timesler/bnp-paribas-cardif-claims-management/xgboost-15-02-2016
library(readr)
library(xgboost)

# Run settings
md <- 11
ss <- 0.96
cs <- 0.45
mc <- 1
np <- 1

cat("Set seed\n")
set.seed(0)

cat("Read the train and test data\n")
train <- read_csv("../input/train.csv")
test  <- read_csv("../input/test.csv")

cat("Sample data for early stopping\n")
h <- sample(nrow(train),1000)

cat("Recode NAs to -997\n")
train[is.na(train)]   <- -997
test[is.na(test)]   <- -997

cat("Get feature names\n")
feature.names <- names(train)[c(3:ncol(train))]

cat("Remove highly correlated features\n")
highCorrRemovals <- c("v8","v23","v25","v36","v37","v46",
                      "v51","v53","v54","v63","v73","v81",
                      "v82","v89","v92","v95","v105","v107",
                      "v108","v109","v116","v117","v118",
                      "v119","v123","v124","v128")
feature.names <- feature.names[!(feature.names %in% highCorrRemovals)]

cat("Replace categorical variables with integers\n")
for (f in feature.names) {
  if (class(train[[f]])=="character") {
    levels <- unique(c(train[[f]], test[[f]]))
    train[[f]] <- as.integer(factor(train[[f]], levels=levels))
    test[[f]]  <- as.integer(factor(test[[f]],  levels=levels))
  }
}

tra<-train[,feature.names]

dval<-xgb.DMatrix(data=data.matrix(tra[h,]),label=train$target[h])
dtrain<-xgb.DMatrix(data=data.matrix(tra[-h,]),label=train$target[-h])

watchlist<-list(val=dval,train=dtrain)

param <- list(  objective           = "binary:logistic", 
                booster             = "gbtree",
                eval_metric         = "logloss",
                eta                 = 0.01,
                max_depth           = md,
                subsample           = ss,
                colsample_bytree    = cs,
                min_child_weight    = mc,
                num_parallel_tree   = np
)

nrounds <- 1500 # CHANGE TO >1500
early.stop.round <- 300

cat("Train model\n")
clf <- xgb.train(   params              = param, 
                    data                = dtrain, 
                    nrounds             = nrounds, 
                    verbose             = 1,  #1
                    early.stop.round    = early.stop.round,
                    watchlist            = watchlist,
                    maximize            = FALSE
)

inputs <- c("nrounds"=clf$bestInd,
            "eta"=param$eta,
            "max_depth"=param$max_depth,
            "subsample"=param$subsample,
            "colsample_bytree"=param$colsample_bytree,
            "min_child_weight"=param$min_child_weight,
            "num_parallel_tree"=param$num_parallel_tree)

cat("Calculate predictions\n")
pred1 <- predict(clf,
                 data.matrix(test[,feature.names]),
                 ntreelimit=clf$bestInd)
pred2 <- predict(clf,
                 dval,
                 ntreelimit=clf$bestInd)

submission <- data.frame(ID=test$ID, PredictedProb=pred1)

LL <- clf$bestScore
cat(paste("Best AUC: ",LL,"\n",sep=""))

cat("Create submission file\n")
time <- format(Sys.time(),"%Y%m%dT%H%M%S")

submission <- submission[order(submission$ID),]
write.csv(submission,
          paste("XGB_",
                paste(as.character(inputs),collapse="_"),
                "_",
                as.character(LL),
                "_",
                time,
                ".csv",sep=""),
          row.names=F)
