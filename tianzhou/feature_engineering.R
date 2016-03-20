library(readr)
library(xgboost)
library(dplyr)
library(e1071)
library(Matrix)
'%nin_v%'<-function(x,y) x[!x %in% y]
'%in_v%'<-function(x,y) x[x %in% y]
group_data <- function(data, degree = 2){
  m <- ncol(data)
  indicies <- combn(1:m, degree)
  dataStr <- apply(indicies, 2, function(s) apply(data[,s], 1, function(x) paste0(x, collapse = "a")))
  dataStr
}

# All infrequent categories into one category
FactorVar <- function(v, lb = 20){
  tbl <- sort(table(v), decreasing = TRUE)
  len <- length(tbl)
  freqEnough <- which(tbl > lb)
  if(length(freqEnough) == 0) return(NULL) # Drop this variable
  k <- max(freqEnough)
  if(k < len){
    vals <- names(tbl[1:k])
    v[! v %in% vals] <- "other"
  } 
  factor(v)
}

train=read_csv("./data/train.csv")
test=read_csv("./data/test.csv")
test['target']<-0.76
###### Group 1 features 
feature.char<-names(train[,sapply(train,is.character)])
feature.int<-names(train[,sapply(train,is.integer)])
feature_all<-c(feature.int,feature.char)
feature_all<-feature_all%nin_v%c("ID","target","dataset")
X_char<-data_all[,feature_all]
X_char$dataset<-NULL
X1 <- group_data(X_char, 1)
XallF <- FactorVar(X1[,1], 0)
for(j in 2:ncol(X1)){
  x <- FactorVar(X1[,j], 0)
  XallF <- data.frame(XallF, x)
}
XallN <- data.frame(lapply(XallF, as.integer))
names(XallN)<-paste0(rep("Group1_N_"),1:23)
names(XallF)<-paste0(rep("Group1_F_"),1:23)
for(i in 1:ncol(XallN)){
  name_c<-paste(names(XallN)[i],"c",sep="")
  name_o<-names(XallN)[i]
  w=XallN%>% dplyr::group_by_(name_o) %>% dplyr::mutate(count=n())
  XallN[,name_c]=w$count
}

set.seed(201602)
train$dataset<-"train"
test$dataset<-"test"
data_all=bind_rows(train,test)
data_all=cbind(data_all,XallN)



##### some selected factor grouped features with 50 sample as threshold
data_all[is.na(data_all)] <- -1

all.features<-names(data_all)[c(3:133)]
for (f in all.features){
  if (class(data_all[[f]])=="character") {
    levels <- unique(data_all[[f]])
    data_all[[f]] <- factor(data_all[[f]], levels=levels)
  }} 
data_all[,"v66_v47"]<-interaction(data_all$v66,data_all$v47,sep=":")
data_all[,"v66_v56"]<-interaction(data_all$v66,data_all$v56,sep=":")
data_all[,"v66_v110"]<-interaction(data_all$v66,data_all$v110,sep=":")
data_all[,"v66_v24"]<-interaction(data_all$v66,data_all$v24,sep=":")
data_all[,"v47_v56"]<-interaction(data_all$v47,data_all$v56,sep=":")
data_all[,"v47_v110"]<-interaction(data_all$v47,data_all$v110,sep=":")
data_all[,"v47_v24"]<-interaction(data_all$v47,data_all$v24,sep=":")
data_all[,"v56_v110"]<-interaction(data_all$v56,data_all$v110,sep=":")
data_all[,"v56_v24"]<-interaction(data_all$v56,data_all$v24,sep=":")
data_all[,"v110_v24"]<-interaction(data_all$v110,data_all$v24,sep=":")

data_all[,"v31_v66"]<-interaction(data_all$v31,data_all$v66,sep=":")
data_all[,"v31_v47"]<-interaction(data_all$v31,data_all$v47,sep=":")
data_all[,"v31_v56"]<-interaction(data_all$v31,data_all$v56,sep=":")
data_all[,"v31_v110"]<-interaction(data_all$v31,data_all$v110,sep=":")
data_all[,"v31_v24"]<-interaction(data_all$v31,data_all$v24,sep=":")

for (f in names(data_all)[c(154:168)]){
  if (class(data_all[[f]])=="factor") {
    data_all[[f]]<-as.character(data_all[[f]])
    w=table(data_all[[f]])
    for(i in 1:dim(data_all)[1]){
      if(w[[data_all[[f]][i]]]<50){
        data_all[[f]][i]<-"minority"
      }
    }
    data_all[[f]]<-as.factor(data_all[[f]])
  }}




####### group_2 features:all char feature+ int feature pairwise grouping and their counts features 
#XallN<-read_csv("feature_group_2_withInt.csv")


X2 <- group_data(X_char, 2)
XallF <- FactorVar(X2[,1], 0)
for(j in 2:ncol(X2)){
  x <- FactorVar(X2[,j], 0)
  XallF <- data.frame(XallF, x)
}
XallN <- data.frame(lapply(XallF, as.integer))
names(XallN)<-paste0(rep("Xall_N_"),1:253)
names(XallF)<-paste0(rep("Xall_F_"),1:253)
for(i in 1:ncol(XallN)){
  name_c<-paste(names(XallN)[i],"c",sep="")
  name_o<-names(XallN)[i]
  w=XallN%>% dplyr::group_by_(name_o) %>% dplyr::mutate(count=n())
  XallN[,name_c]=w$count
}

name_list<-names(XallN)
for(i in 1:ncol(XallN)){
  name_c<-paste(names(XallN)[i],"c",sep="")
  name_o<-names(XallN)[i]
  w=XallN%>% dplyr::group_by_(name_o) %>% dplyr::mutate(count=n())
  XallN[,name_c]=w$count
}
data_all=cbind(data_all,XallN)

####### double frequent features features:all char feature+ int feature pairwise grouping and their counts features 
pairs <- combn(names(X_char), 2, simplify=FALSE)
for(i in 1:length(pairs)){
  feature1 <- pairs[[i]][1]
  feature2 <- pairs[[i]][2]
  X_char[,paste0(feature1,feature2)]<-paste(X_char[[feature1]],X_char[[feature2]],sep="_a")
  X_char[,paste0(feature2,feature1)]<-paste(X_char[[feature2]],X_char[[feature1]],sep="_a")
  
  cols=c(feature1,feature2)
  dots<-lapply(cols,as.symbol)
  w1<-X_char %>%
    group_by_(.dots=dots)%>%
    summarise(n=n()) %>%
    mutate(freq = n/sum(n))
  w1[,paste0(feature1,feature2)]<- paste(w1[[feature1]],w1[[feature2]],sep="_a")
  w1[[feature1]]<-NULL
  w1[[feature2]]<-NULL
  w1[["n"]]<-NULL
  X_char<-X_char %>% left_join(w1,by=paste0(feature1,feature2))
  len<-which(names(X_char)=='freq')
  names(X_char)[len]<-paste0(feature1,feature2,"freq")
  
  cols=c(feature2,feature1)
  dots<-lapply(cols,as.symbol)
  w2<-X_char %>%
    group_by_(.dots=dots)%>%
    summarise(n=n()) %>%
    mutate(freq = n/sum(n))
  w2[,paste0(feature2,feature1)]<- paste(w2[[feature2]],w2[[feature1]],sep="_a")
  w2[[feature1]]<-NULL
  w2[[feature2]]<-NULL
  w2[["n"]]<-NULL
  X_char<-X_char %>% left_join(w2,by=paste0(feature2,feature1))
  len<-which(names(X_char)=='freq')
  names(X_char)[len]<-paste0(feature2,feature1,"freq")
  X_char[,paste0(feature1,feature2)]<-NULL
  X_char[,paste0(feature2,feature1)]<-NULL
}
##### total features
data_all=cbind(data_all,X_char[,names(X_char)[24:529]])

train<- data_all %>% dplyr::filter(dataset %in% c("train"))
test<- data_all %>% dplyr::filter(dataset %in% c("test"))
train$dataset<-NULL
test$dataset<-NULL
write.csv(train, file='train_tian.csv', quote=FALSE,row.names=FALSE)
write.csv(test, file='test_tian.csv', quote=FALSE,row.names=FALSE)
