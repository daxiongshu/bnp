library(h2o)
library(readr)
h2o.init(nthreads=-1,max_mem_size = '20G')
### load both files in using H2O's parallel import
train<-read_csv("../input/tr.csv")  
test<-read_csv("../input/va.csv")     
train$target<-as.factor(train$target)
x<-names(which(sapply(train,is.numeric)==TRUE))
train<-as.h2o(train)
test<-as.h2o(test)


model<-h2o.deeplearning(x=x,
                         y="target",
                         training_frame = train,
                         nfolds = 3,
                         stopping_rounds = 3,
                         epochs = 20,
                         overwrite_with_best_model = TRUE,
                         activation = "RectifierWithDropout",
                         input_dropout_ratio = 0.2,
                         hidden = c(100,100),
                         l1 = 1e-4,
                         loss = "CrossEntropy",
                         distribution = "bernoulli",
                         stopping_metric = "logloss"
                         
                         
)
### get predictions against the test set and create submission file
p<-as.data.frame(h2o.predict(model,test))
testIds<-as.data.frame(test$ID)
submission<-data.frame(cbind(testIds,p$p1))
colnames(submission)<-c("ID","PredictedProb")
write.csv(submission,"h2onn1cv.csv",row.names=F)
