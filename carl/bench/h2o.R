library(h2o)
h2o.init(nthreads=-1,max_mem_size = '8G')
### load both files in using H2O's parallel import
train<-h2o.uploadFile("../input/train.csv",destination_frame = "train.hex")  
test<-h2o.uploadFile("../input/test.csv",destination_frame = "test.hex")     
train$target<-as.factor(train$target)
splits<-h2o.splitFrame(train,0.9,destination_frames = c("trainSplit","validSplit"),seed=111111111)
gbm<-h2o.gbm(
  x = 3:133,
  y=2,
  training_frame = splits[[1]],
  validation_frame = splits[[2]],
  ntrees = 3000,                    ## let stopping criteria dictate the number of trees
  stopping_rounds = 1,              ## wait until the last round is worse than the previous
                                    ##  this seems low because scoring is not on every tree by default
                                    ##  If that is desired, you can turn on score_each_iteration
                                    ## (and then possibly increase stopping)
  stopping_tolerance = 0,
  max_depth = 5,
  learn_rate = 0.03,
  sample_rate = 0.8,                ## 80% row sampling
  col_sample_rate = 0.7,            ## 70% columns
  seed = 222222222,
  model_id = "baseGbm")

### look at some information about the model
summary(gbm)
h2o.logloss(gbm,valid=T)

### get predictions against the test set and create submission file
p<-as.data.frame(h2o.predict(gbm,test))
testIds<-as.data.frame(test$ID)
submission<-data.frame(cbind(testIds,p$p1))
colnames(submission)<-c("ID","PredictedProb")
write.csv(submission,"h2o.csv",row.names=F)
