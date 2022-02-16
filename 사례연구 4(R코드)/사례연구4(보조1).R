### 사례연구 4 남원식 #
#1.위스콘신유방암 데이터셋을대상으로 분류기법 2개를  적용하여 기법별  결과를 비교하시오. 
#-종속변수는diagnosis: Benign(양성), Malignancy(악성)
#### 의사결정 트리 기법 ####
rm(list = ls())
install.packages("party")
install.packages("caret")
library(party)
library(caret)
table(wisc$diagnosis2)

setwd("C:/Rwork/data")
wisc <- read.csv("wisc_bc_data.csv")
#불필요 NA값만들어있는 컬럼 제거
wisc <- subset(wisc, select = -X)
wisc
table(wisc$diagnosis)
str(wisc)
#요인형으로 변환
wiscf <- factor(wisc$diagnosis)
wisc$diagnosis2 <- wiscf
wisc$diagnosis <- NULL
# 트레이닝, 테스트셋 
set.seed(1)
idx <- sample(1:nrow(wisc),nrow(wisc)*0.7)
train_wisc <- wisc[idx,]
test_wisc <-  wisc[-idx,]
#포물러 생성 
wisc_formula <- diagnosis2 ~ + radius_mean + texture_mean + perimeter_mean +
  area_mean + smoothness_mean + compactness_mean + concavity_mean + concave.points_mean +
  symmetry_mean + fractal_dimension_mean + radius_se + texture_se + perimeter_se + 
  area_se + smoothness_se + compactness_se + concavity_se + concave.points_se +
  symmetry_se + fractal_dimension_se + radius_worst + texture_worst + perimeter_worst +
  area_worst + smoothness_worst + compactness_worst + concavity_worst + concave.points_worst +
  symmetry_worst + fractal_dimension_worst 
# 의사 사 결정 나무 형성
wisc_tree1 <- ctree(wisc_formula, data = train_wisc)
wisc_tree1
plot(wisc_tree1,type = "simple")
# 테스트셋 검정 오분류율 테스트
predicted_wisc <- predict(wisc_tree1,test_wisc)
plot(predicted_wisc)
text(predicted_wisc)
confusionMatrix(predicted_wisc,test_wisc$diagnosis2)
# 0.9006 민감도 0.9320 특정도 0.8529
# 암환자의 데이터 같은경우는 양성보다 악성을 잘못 판단하였을 경우 문제가 더 커지기때문에
# 특정도가 더 중요한 것으로 판단이됨. 
# 두수치 모두 높은것이 가장 바람직하며 실제로 분류율이 높은것으로 판단이된다.


#### 베이지안 #####
rm(list = ls())
install.packages("e1071")
install.packages("caret")
library(e1071)
library(caret)

setwd("C:/Rwork/data")
wisc <- read.csv("wisc_bc_data.csv")
#불필요 NA값만들어있는 컬럼 제거
wisc <- subset(wisc, select = -X)
wisc
#요인형으로 변환
wiscf <- factor(wisc$diagnosis)
wisc$diagnosis2 <- wiscf
wisc$diagnosis <- NULL
#포뮬러
wisc_formula <- diagnosis2 ~ + radius_mean + texture_mean + perimeter_mean +
  area_mean + smoothness_mean + compactness_mean + concavity_mean + concave.points_mean +
  symmetry_mean + fractal_dimension_mean + radius_se + texture_se + perimeter_se + 
  area_se + smoothness_se + compactness_se + concavity_se + concave.points_se +
  symmetry_se + fractal_dimension_se + radius_worst + texture_worst + perimeter_worst +
  area_worst + smoothness_worst + compactness_worst + concavity_worst + concave.points_worst +
  symmetry_worst + fractal_dimension_worst 
# 트레이닝, 테스트셋 
set.seed(1)
idx <- createDataPartition(y=wisc$diagnosis,p=0.7,list=FALSE)
train_wisc <- wisc[idx,]
test_wisc <-  wisc[-idx,]
# 베이지안
bayes_wisc_tr <- naiveBayes(diagnosis2~.,data = train_wisc)
# 테스트셋 검정
pre_bayes_wisc <- predict(bayes_wisc_tr,test_wisc,type = "class")
table(pre_bayes_wisc,test_wisc$diagnosis)
bayes_wisc_fin <- as.factor(test_wisc$diagnosis)
confusionMatrix(pre_bayes_wisc,bayes_wisc_fin)
# 정확도 분류율 93.53  sensitivity = 0.9626 , specifictity = 0.8889
# 똑같은 데이터로 분류를 진행했을때 니아브 베이지안 기법이 의사결정 나무기법에 비해서
# 더 높은 분류율을 가지고 있음을 알수있음.


##### 로지스틱 회귀함수 ####
rm(list = ls())
install.packages("ROCR")
library(VGAM)  #다중로지스틱회귀함수 package
library(car)
library(lmtest)
library(ROCR)
setwd("C:/Rwork/data")
wisc <- read.csv("wisc_bc_data.csv")
str(wisc)
wisc <- subset(wisc, select = -X)
#요인형으로 변환
wiscf <- factor(wisc$diagnosis)
wisc$diagnosis2 <- wiscf
wisc$diagnosis <- NULL
#트레이닝, 테스트셋 생성
idx <- sample(1:nrow(wisc),nrow(wisc)*0.7)
train_wisc <- wisc[idx,]
test_wisc <-  wisc[-idx,]
#포물러 생성 
wisc_formula <- diagnosis2 ~ + radius_mean + texture_mean + perimeter_mean +
  area_mean + smoothness_mean + compactness_mean + concavity_mean + concave.points_mean +
  symmetry_mean + fractal_dimension_mean + radius_se + texture_se + perimeter_se + 
  area_se + smoothness_se + compactness_se + concavity_se + concave.points_se +
  symmetry_se + fractal_dimension_se + radius_worst + texture_worst + perimeter_worst +
  area_worst + smoothness_worst + compactness_worst + concavity_worst + concave.points_worst +
  symmetry_worst + fractal_dimension_worst 
#로지스틱 회귀 분류
wisc_model <- glm(wisc_formula,data = train_wisc,family = binomial)
#학습 로지스틱 모델의 test셋 예측
pro_wisc_model <- predict(wisc_model,test_wisc,type = "response")
pro_wisc_model
#각행의 합을 확인해서 1로 떨어지면 일치
apply(pro_wisc_model,1,FUN=sum)
table(pro_wisc_model)
table(test_wisc$diagnosis)
wtable <- table(test_wisc$diagnosis,pro_wisc_model)
wtable
######################에러 스킵 #######로지스틱회귀분석시 변수가너무많으면 오류 있다고함########
####2.Abalone Data 데이터셋을 대상으로 전복의 나이를 예측하고자  한다.####
# 예측기법   2개를    적용하여 기법별    결과를    비교하시오.

#### 의사결정 트리 ####
rm(list = ls())
install.packages("party")
install.packages("caret")
library(party)
library(caret)
library(ggplot2)
setwd("C:/Rwork/data")
abalone <- read.csv("abalone.csv",header = T)
max(abalone$Rings)
# rings 범주화
abalone$Rings2[abalone$Rings>=0 & abalone$Rings<=6] = 1
abalone$Rings2[abalone$Rings>=7 & abalone$Rings<=12] = 2
abalone$Rings2[abalone$Rings>=13 & abalone$Rings<=18] = 3
abalone$Rings2[abalone$Rings>=19 & abalone$Rings<=24] = 4
abalone$Rings2[abalone$Rings>=25 & abalone$Rings<=29] = 5
abalone$Rings = NULL
abalone$Rings2 = factor(abalone$Rings)
abalone$Rings2 <- factor(abalone$Rings2)
table(abalone$Rings2)
abalone$Sex <- factor(abalone$Sex)
is.na(abalone)
# 트레이닝,검정셋
idx <- sample(1:nrow(abalone),nrow(abalone)*0.7)
train_abalone <- abalone[idx,]
test_abalone <-  abalone[-idx,]
# 포뮬러 생성
aba_formula <- Rings2 ~ Sex + Length + Diameter + Height + Whole.weight + Shucked.weight + 
Viscera.weight + Shell.weight    
# 의사결정나무 생성
aba_tree <- ctree(aba_formula, data=train_abalone)
aba_tree
plot(aba_tree)
# 테스트셋 검정
predicted_aba <- predict(aba_tree,test_abalone)
plot(predicted_aba)
text(predicted_aba)
confusionMatrix(predicted_aba,test_abalone$Rings2)
#예측하기

#### 랜덤포레스트 방식 ####
install.packages("randomForest")
library(randomForest)
setwd("C:/Rwork/data")
abalone <- read.csv("abalone.csv",header = T)
# rings 범주화
#abalone$Rings2[abalone$Rings>=0 & abalone$Rings<=6] = 1
#abalone$Rings2[abalone$Rings>=7 & abalone$Rings<=12] = 2
#abalone$Rings2[abalone$Rings>=13 & abalone$Rings<=18] = 3
#abalone$Rings2[abalone$Rings>=19 & abalone$Rings<=24] = 4
#abalone$Rings2[abalone$Rings>=25 & abalone$Rings<=29] = 5
#abalone$Rings = NULL
abalone$Rings <- factor(abalone$Rings)
#table(abalone$Rings2)
#abalone$Sex <- factor(abalone$Sex)
# 트레이닝,검정셋
#set.seed(1)
idx <- sample(1:nrow(abalone),nrow(abalone)*0.7)
train_abalone <- abalone[idx,]
test_abalone <-  abalone[-idx,]
# 포뮬러 생성
aba_formula <- Rings ~ Sex + Length + Diameter + Height + Whole.weight + Shucked.weight + 
  Viscera.weight + Shell.weight    
abalone_rf = randomForest(aba_formula,data = abalone, importance=T)
importance(abalone_rf)
varImpPlot(abalone_rf)
# 테스트셋 검정
aba_predict_rf <- predict(abalone_rf,newdata =test_abalone$Rings)
aba_predict_rf
table(aba_predict_rf,test_abalone$Rings)
confusionMatrix(aba_predict_rf,test_abalone$Rings)

####선형회귀  ####
setwd("C:/Rwork/data")
abalone <- read.csv("abalone.csv",header = T)
# 트레이닝,검정셋
set.seed(1)
idx <- sample(1:nrow(abalone),nrow(abalone)*0.7)
train_abalone <- abalone[idx,]
test_abalone <-  abalone[-idx,]
# 포뮬러 생성
aba_formula <- Rings ~ Sex + Length + Diameter + Height + Whole.weight + Shucked.weight + 
  Viscera.weight + Shell.weight    
aba_lm <- lm(aba_formula, data = abalone)
plot(aba_lm)
aba_lm_pr <- predict(aba_lm, data = test_abalone$Rings)
confusionMatrix(abalone$Rings,aba_lm_pr) 
cor(aba_lm_pr,abalone$Rings)#0.733406 상관계수
#예측값을 구하기위한 회귀방정식


####3.iris데이터에서 species 컬럼 데이터를 제거한 후 k-means clustering를 실행하고 시각화하시오####
library(ggplot2)
install.packages("mclust")
install.packages("corrgram")
install.packages("NbClust")
library(mclust)
library(corrgram)
data("iris")
iris$Species <- NULL
# 최적의 군집수 찾기
library(NbClust)
set.seed(1234)
iris_c <- NbClust(iris, min.nc=2, max.nc=15, method="kmeans")
table(iris_c$Best.n[1,])
# 2~ 3개의 군집수가 적장
# Species 제외
iris <- subset(iris, select = -Species)
# k-means cluster
iris_kmean <- kmeans(iris,4) #군집 6개 분류
iris_kmean
str(iris_kmean)
# K-means cluster 플로팅
plot(iris[c("Sepal.Length", "Sepal.Width")], col=iris_kmean$cluster)
points(iris_kmean$centers[, c("Sepal.Length", "Sepal.Width")], col=1:4, pch=8, cex=2)
plot(iris[c("Petal.Length", "Petal.Width")], col=iris_kmean$cluster)
points(iris_kmean$centers[, c("Petal.Length", "Petal.Width")], col=1:4, pch=8, cex=2)

# 유사성이 있을수록 클러스터로부터의 길이가 짧다 .
# 군집수를 정한 이유 ? 를 설명해야함

