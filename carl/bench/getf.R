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
te<-test[,feature.names]


write.csv(tra,"train_clean1.csv",row.names=F)
write.csv(te,"test_clean1.csv",row.names=F)
