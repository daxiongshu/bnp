library(ggplot2) # Data visualization
library(readr) # CSV file I/O, e.g. the read_csv function
library(randomForest)
library(mice)
#library(VIM)
library(ggplot2)
library(lattice)
#library(car)
library(ROCR)



# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

system("ls ../input")

#read the training data
newdata <- read_csv("../input/train.csv")

#finding only charracter and only continous columns
cat.var.names <- colnames(newdata)[sapply(newdata, is.factor)]
cont.var.names <- colnames(newdata)[sapply(newdata, is.numeric)]

newdataNum <- newdata[,cont.var.names]
newdataChar <- newdata[,cat.var.names]


# take a smaller dataset to check imputation
sampleRow<-sort(sample(nrow(newdataNum),nrow(newdataNum)*.01))
newdataSampleNum <- newdataNum[sampleRow,]

tempData <- mice(newdataSampleNum,meth='pmm', maxit=1, m=1) # using as 1 to save processing in this version.
# to extract the data set for model building, tempdata has a different class. so storing in compNum
compNum<-complete(tempData) # we can choose which data set from the m created, m=2 in mice commnad is the param.

#fit <- glm(target ~ ., data =compNum, family=binomial)
#summary(fit)

#v126,v108,103,v72,v1,v2,v9, v11, v17,  v25, v29,v53,v62, v63, why 129 is NA, 
#fit <- glm(target ~ v1+v2+v9+v11+v17+v25+v29+v53+v62+v63+v72+v108+v103+v126, data =compNum, family=binomial)
#summary(fit)

fit <- glm(target ~ v62+v72, data =compNum, family=binomial)
summary(fit)

# processing character columns,

#sampleRowChar<-sort(sample(nrow(newdataChar),nrow(newdataChar)*.01))
#newdataSampleChar <- newdataChar[sampleRowChar,]
#targetChar <- newdata[sampleRowChar,"target"]
#newdataSampleChar <- cbind(newdataSampleChar,targetChar)
#attach(newdataSampleChar)
#fitChar <- glm(targetChar ~ v22+v24+v30+v31+v47+v52+v56+v66+v71+v74+v79+v91+v112+v113 , data =newdataSampleChar, family=binomial)
#summary(fitChar)
#detach(newdataSampleChar)
#with significant variables create a new data set having both char and continous variables

dataSig <- newdata[ , c("ID", "target" , "v62", "v72", "v22", "v24", "v30", 
                        "v31", "v47", "v52", "v56", "v66", "v71", "v74", "v75" ,
                        "v79", "v91" ,"v112", "v113")]

# There are no NA in dataSig
attach(dataSig)

#treat v22 with dummy variables
dataSig$v22_AGDF = ifelse(v22 == "AGDF", 1,0)
dataSig$v22_GBS = ifelse(v22 == "GBS", 1,0)
dataSig$v22_HZE = ifelse(v22 == "HZE", 1,0)
dataSig$v22_MNZ = ifelse(v22 == "MNZ", 1,0)
dataSig$v22_PTO = ifelse(v22 == "PTO", 1,0)
dataSig$v22_PWR = ifelse(v22 == "PWR", 1,0)
dataSig$v22_QKI = ifelse(v22 == "QKI", 1,0)
dataSig$v22_ROZ = ifelse(v22 == "ROZ", 1,0)
dataSig$v22_YGJ = ifelse(v22 == "YGJ", 1,0)
dataSig$v22_YOD = ifelse(v22 == "YOD", 1,0)
dataSig$v22_ADDF = ifelse(v22 == "ADDF", 1,0)
dataSig$v22_AJQ = ifelse(v22 == "AJQ", 1,0)
dataSig$v22_HDD = ifelse(v22 == "HDD", 1,0)
dataSig$v22_NRT = ifelse(v22 == "NRT", 1,0)
dataSig$v22_PFR = ifelse(v22 == "PFR", 1,0)
dataSig$v22_QVR = ifelse(v22 == "QVR", 1,0)
dataSig$v22_VVI = ifelse(v22 == "VVI", 1,0)
dataSig$v22_VZF = ifelse(v22 == "VZF", 1,0)
dataSig$v22_WNI = ifelse(v22 == "WNI", 1,0)
dataSig$v22_YEP = ifelse(v22 == "YEP", 1,0)

#treat v24 with dummy variables
dataSig$v24_A = ifelse(v24 == "A", 1,0)
dataSig$v24_B = ifelse(v24 == "B", 1,0)
dataSig$v24_C = ifelse(v24 == "C", 1,0)
dataSig$v24_D = ifelse(v24 == "D", 1,0)
dataSig$v24_E = ifelse(v24 == "E", 1,0)

#treat v30 with dummy variables
dataSig$v30_A = ifelse(v30 == "A", 1,0)
dataSig$v30_B = ifelse(v30 == "B", 1,0)
dataSig$v30_C = ifelse(v30 == "C", 1,0)
dataSig$v30_D = ifelse(v30 == "D", 1,0)
dataSig$v30_E = ifelse(v30 == "E", 1,0)
dataSig$v30_F = ifelse(v30 == "F", 1,0)
dataSig$v30_G = ifelse(v30 == "G", 1,0)

#treat v31 with dummy variables
dataSig$v31_A = ifelse(v31 == "A", 1,0)
dataSig$v31_B = ifelse(v31 == "B", 1,0)
dataSig$v31_C = ifelse(v31 == "C", 1,0)

#treat v47 with dummy variables
dataSig$v47_A = ifelse(v47 == "A", 1,0)
dataSig$v47_B = ifelse(v47 == "B", 1,0)
dataSig$v47_C = ifelse(v47 == "C", 1,0)
dataSig$v47_D = ifelse(v47 == "D", 1,0)
dataSig$v47_E = ifelse(v47 == "E", 1,0)
dataSig$v47_F = ifelse(v47 == "F", 1,0)
dataSig$v47_G = ifelse(v47 == "G", 1,0)
dataSig$v47_H = ifelse(v47 == "H", 1,0)
dataSig$v47_I = ifelse(v47 == "I", 1,0)
dataSig$v47_I = ifelse(v47 == "J", 1,0)

#treat v56 with dummy variables
dataSig$v56_AS = ifelse(v56 == "AS", 1,0)
dataSig$v56_AW = ifelse(v56 == "AW", 1,0)
dataSig$v56_BW = ifelse(v56 == "BW", 1,0)
dataSig$v56_BZ = ifelse(v56 == "BZ", 1,0)
dataSig$v56_CN = ifelse(v56 == "CN", 1,0)
dataSig$v56_CY = ifelse(v56 == "CY", 1,0)
dataSig$v56_DI = ifelse(v56 == "DI", 1,0)
dataSig$v56_DO = ifelse(v56 == "DO", 1,0)
dataSig$v56_DP = ifelse(v56 == "DP", 1,0)
dataSig$v56_P = ifelse(v56 == "P", 1,0)

#treat v66 with dummy variables
dataSig$v66_A = ifelse(v66 == "A", 1,0)
dataSig$v66_B = ifelse(v66 == "B", 1,0)
dataSig$v66_C = ifelse(v66 == "C", 1,0)

#treat v74 with dummy variables
dataSig$v74_A = ifelse(v74 == "A", 1,0)
dataSig$v74_B = ifelse(v74 == "B", 1,0)
dataSig$v74_C = ifelse(v74 == "C", 1,0)


#treat v75 with dummy variables
dataSig$v75_A = ifelse(v75 == "A", 1,0)
dataSig$v75_B = ifelse(v75 == "B", 1,0)
dataSig$v75_C = ifelse(v75 == "C", 1,0)
dataSig$v75_D = ifelse(v75 == "D", 1,0)


#treat v79 with dummy variables
dataSig$v79_B = ifelse(v79 == "B", 1,0)
dataSig$v79_C = ifelse(v79 == "C", 1,0)
dataSig$v79_D = ifelse(v79 == "D", 1,0)
dataSig$v79_E = ifelse(v79 == "E", 1,0)
dataSig$v79_H = ifelse(v79 == "H", 1,0)
dataSig$v79_I = ifelse(v79 == "I", 1,0)
dataSig$v79_K = ifelse(v79 == "K", 1,0)
dataSig$v79_M = ifelse(v79 == "M", 1,0)
dataSig$v79_O = ifelse(v79 == "O", 1,0)
dataSig$v79_P = ifelse(v79 == "P", 1,0)

#treat v112 with dummy variables
dataSig$v112_A = ifelse(v112 == "A", 1,0)
dataSig$v112_D = ifelse(v112 == "D", 1,0)
dataSig$v112_E = ifelse(v112 == "E", 1,0)
dataSig$v112_F = ifelse(v112 == "F", 1,0)
dataSig$v112_H = ifelse(v112 == "H", 1,0)
dataSig$v112_I = ifelse(v112 == "I", 1,0)
dataSig$v112_L = ifelse(v112 == "L", 1,0)
dataSig$v112_N = ifelse(v112 == "N", 1,0)
dataSig$v112_P = ifelse(v112 == "P", 1,0)
dataSig$v112_U = ifelse(v112 == "U", 1,0)

#treat v113  with dummy variables
dataSig$v113_AC = ifelse(v113 == "AC", 1,0)
dataSig$v113_AF = ifelse(v113 == "AF", 1,0)
dataSig$v113_AG = ifelse(v113 == "AG", 1,0)
dataSig$v113_G = ifelse(v113 == "G", 1,0)
dataSig$v113_I = ifelse(v113 == "I", 1,0)
dataSig$v113_M = ifelse(v113 == "M", 1,0)
dataSig$v113_P = ifelse(v113 == "P", 1,0)
dataSig$v113_T = ifelse(v113 == "T", 1,0)
dataSig$v113_V = ifelse(v113 == "V", 1,0)
dataSig$v113_X = ifelse(v113 == "X", 1,0)

attach(dataSig)
fitAll <- glm(target ~  v62+
                v72 +
                #          v22_AGDF +
                #          v22_MNZ +
                #          v22_WNI +
                v24_B +
                v24_C +
                v24_E +
                #          v30_B +
                v30_C +
                #          v30_E +
                #          v30_F +
                v31_A +
                v31_C +
                v47_I +
                v56_AS +
                v56_AW +
                v56_BW +
                v56_BZ +
                v56_CY +
                v56_DO +
                #          v56_DP +
                v56_P +
                v66_A +
                v66_B +
                v79_B +
                v79_C +
                v79_E +
                v79_H +
                v79_P +
                v112_F +
                v113_AG +
                v113_G +
                v113_I +
                v113_M 
              ,
              data =dataSig, family=binomial)
summary(fitAll)

#checking for multi collinearity in the model
#vif(fitAll)


#checking for confidence intervals in the model
#confint(fitAll)

#validate the model using confusion matrix
predicted <- fitAll$fitted.values
#head(predicted)
#predBkt <- ifelse(predicted > 0.65, "G", "B")
#table(predBkt,target)

# use ROCR package to find AUC of the the curve
#pred <- prediction(predicted, target)
#perf <-performance(pred, "tpr", "fpr")
#plot(perf)
#auc <- performance(pred, "auc")
#auc <-unlist(slot(auc, "y.values"))
#auc #auc of 69% is assumed good. 

detach(dataSig)

# try hands on the test data set
testData<-read.csv("../input/test.csv", header=TRUE)
testSig <- testData[ , c("ID" , "v62", "v72", "v24", "v30", 
                         "v31", "v47", "v56", "v66",
                         "v79","v112", "v113")]


attach(testSig)

#treat v24 with dummy variables
testSig$v24_A = ifelse(v24 == "A", 1,0)
testSig$v24_B = ifelse(v24 == "B", 1,0)
testSig$v24_C = ifelse(v24 == "C", 1,0)
testSig$v24_D = ifelse(v24 == "D", 1,0)
testSig$v24_E = ifelse(v24 == "E", 1,0)

#treat v30 with dummy variables
testSig$v30_A = ifelse(v30 == "A", 1,0)
testSig$v30_B = ifelse(v30 == "B", 1,0)
testSig$v30_C = ifelse(v30 == "C", 1,0)
testSig$v30_D = ifelse(v30 == "D", 1,0)
testSig$v30_E = ifelse(v30 == "E", 1,0)
testSig$v30_F = ifelse(v30 == "F", 1,0)
testSig$v30_G = ifelse(v30 == "G", 1,0)

#treat v31 with dummy variables
testSig$v31_A = ifelse(v31 == "A", 1,0)
testSig$v31_B = ifelse(v31 == "B", 1,0)
testSig$v31_C = ifelse(v31 == "C", 1,0)

#treat v47 with dummy variables
testSig$v47_A = ifelse(v47 == "A", 1,0)
testSig$v47_B = ifelse(v47 == "B", 1,0)
testSig$v47_C = ifelse(v47 == "C", 1,0)
testSig$v47_D = ifelse(v47 == "D", 1,0)
testSig$v47_E = ifelse(v47 == "E", 1,0)
testSig$v47_F = ifelse(v47 == "F", 1,0)
testSig$v47_G = ifelse(v47 == "G", 1,0)
testSig$v47_H = ifelse(v47 == "H", 1,0)
testSig$v47_I = ifelse(v47 == "I", 1,0)
testSig$v47_I = ifelse(v47 == "J", 1,0)

#treat v56 with dummy variables
testSig$v56_AS = ifelse(v56 == "AS", 1,0)
testSig$v56_AW = ifelse(v56 == "AW", 1,0)
testSig$v56_BW = ifelse(v56 == "BW", 1,0)
testSig$v56_BZ = ifelse(v56 == "BZ", 1,0)
testSig$v56_CN = ifelse(v56 == "CN", 1,0)
testSig$v56_CY = ifelse(v56 == "CY", 1,0)
testSig$v56_DI = ifelse(v56 == "DI", 1,0)
testSig$v56_DO = ifelse(v56 == "DO", 1,0)
testSig$v56_DP = ifelse(v56 == "DP", 1,0)
testSig$v56_P = ifelse(v56 == "P", 1,0)

#treat v66 with dummy variables
testSig$v66_A = ifelse(v66 == "A", 1,0)
testSig$v66_B = ifelse(v66 == "B", 1,0)
testSig$v66_C = ifelse(v66 == "C", 1,0)

#treat v79 with dummy variables
testSig$v79_B = ifelse(v79 == "B", 1,0)
testSig$v79_C = ifelse(v79 == "C", 1,0)
testSig$v79_D = ifelse(v79 == "D", 1,0)
testSig$v79_E = ifelse(v79 == "E", 1,0)
testSig$v79_H = ifelse(v79 == "H", 1,0)
testSig$v79_I = ifelse(v79 == "I", 1,0)
testSig$v79_K = ifelse(v79 == "K", 1,0)
testSig$v79_M = ifelse(v79 == "M", 1,0)
testSig$v79_O = ifelse(v79 == "O", 1,0)
testSig$v79_P = ifelse(v79 == "P", 1,0)

#treat v112 with dummy variables
testSig$v112_A = ifelse(v112 == "A", 1,0)
testSig$v112_D = ifelse(v112 == "D", 1,0)
testSig$v112_E = ifelse(v112 == "E", 1,0)
testSig$v112_F = ifelse(v112 == "F", 1,0)
testSig$v112_H = ifelse(v112 == "H", 1,0)
testSig$v112_I = ifelse(v112 == "I", 1,0)
testSig$v112_L = ifelse(v112 == "L", 1,0)
testSig$v112_N = ifelse(v112 == "N", 1,0)
testSig$v112_P = ifelse(v112 == "P", 1,0)
testSig$v112_U = ifelse(v112 == "U", 1,0)

#treat v113  with dummy variables
testSig$v113_AC = ifelse(v113 == "AC", 1,0)
testSig$v113_AF = ifelse(v113 == "AF", 1,0)
testSig$v113_AG = ifelse(v113 == "AG", 1,0)
testSig$v113_G = ifelse(v113 == "G", 1,0)
testSig$v113_I = ifelse(v113 == "I", 1,0)
testSig$v113_M = ifelse(v113 == "M", 1,0)
testSig$v113_P = ifelse(v113 == "P", 1,0)
testSig$v113_T = ifelse(v113 == "T", 1,0)
testSig$v113_V = ifelse(v113 == "V", 1,0)
testSig$v113_X = ifelse(v113 == "X", 1,0)

testSig$targetDummy <- runif(nrow(testSig), 0,1)


attach(testSig)
#model new 1  
fitAllTest <- glm(targetDummy ~  v62+
                    v72 +
                    v24_B +
                    v24_C +
                    v24_E +
                    v30_C +
                    v31_A +
                    v31_C +
                    v47_I +
                    v56_AS +
                    v56_AW +
                    v56_BW +
                    v56_BZ +
                    v56_CY +
                    v56_DO +
                    v56_P +
                    v66_A +
                    v66_B +
                    v79_B +
                    v79_C +
                    v79_E +
                    v79_H +
                    v79_P +
                    v112_F +
                    v113_AG +
                    v113_G +
                    v113_I +
                    v113_M 
                  , 
                  data =testSig, family=binomial)
head(testSig)

summary(fitAllTest)

PredictedProb <- fitAllTest$fitted.values
submission <- cbind(ID,PredictedProb)

write.csv(submission, file="glm1.csv", row.names = FALSE)
