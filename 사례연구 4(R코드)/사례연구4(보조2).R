# 사례연구4 이순규

# 1. 위스콘신 유방암 데이터셋을 대상으로 분류기법 2개를 적용하여 기법별 결과를 비교하시오.
# -종속변수는diagnosis: Benign(양성), Malignancy(악성)

# 데이터 샘플링
rm(list = ls())
getwd()
setwd('D:/')
wisdata <- read.csv('wisc_bc_data.csv', header = T)
head(wisdata)
wisdata_df <- wisdata[-1] # 필요없는 id 컬럼 삭제
wisdata_df <- wisdata_df[-32] # 필요없는 x컬럼 삭제
head(wisdata_df); str(wisdata_df) # 데이터 확인

# 학습데이터, 검정데이터 셋으로 분리
set.seed(1) # 매번 값을 변동 안시키기 위해 설정
idx <- sample(1:nrow(wisdata_df), 0.7*nrow(wisdata_df))
train <- wisdata_df[idx,]
test <- wisdata_df[-idx,]
dim(train); dim(test)

# 1-1 나이브 베이즈 모형
library(caret)
library(e1071)

wiscon_naive <- naiveBayes(train, train$diagnosis, laplace = 1)  # 모델링
wiscon_naive_pred <- predict(wiscon_naive, test, type = "class")  # 모델로 예측
str(wiscon_naive_pred)
str(test$diagnosis)
result <- confusionMatrix(wiscon_naive_pred, as.factor(test$diagnosis)) 
# test$diagnosis as.factor변환 후 코딩 / 이유 : Error: `data` and `reference` should be factors with the same levels.
result
# Accuracy : 0.9649 / 96% 분류정확도

# 1-2 의사결정 나무 분류기법
install.packages('rpart')
install.packages('rpart.plot')
library(rpart)
library(rpart.plot)

wiscon_rpart <- rpart(diagnosis ~ ., data = train)
rpart.plot(wiscon_rpart)  # 트리 시각화

wiscon_pred <- predict(wiscon_rpart, test, type = 'class')  # 평가

result2 <- confusionMatrix(wiscon_pred, as.factor(test$diagnosis))
result2

# Accuracy : 0.8889 / 88% 분류정확도

# 1-3 인공신경망 분류기법
# install.packages('nnet')
# library(nnet)
# head(train); head(test)
# nrow(train); nrow(test)

# wiscon_net = nnet(diagnosis ~ ., train, size = 1)
# 입력 변수의 값들이 일정하지 않거나 값이 큰 경우에는 신경망 모델이 정상적으로 만들어지지
# 않기 때문에 입력 변수를 대상으로 정규화 과정이 필요하다.

# train$diagnosis[train$diagnosis == 'B'] <- 1
# train$diagnosis[train$diagnosis == 'M'] <- 2
# test$diagnosis[test$diagnosis == 'B'] <- 1
# test$diagnosis[test$diagnosis == 'M'] <- 2
# train$diagnosis <- as.numeric(train$diagnosis) # 캐릭터값에서 변경
# test$diagnosis <- as.numeric(test$diagnosis)
# head(train); head(test)
# normal <- function(x){
#   return((x - min(x)) / (max(x) - min(x)))}

# wiscon_train <- as.data.frame(lapply(train, normal))
# wiscon_test <- as.data.frame(lapply(test, normal))

# wiscon_nnet <- nnet(diagnosis ~ ., wiscon_train, size = 2)
# wiscon_nnet_pred <- predict(wiscon_nnet, wiscon_test, type = 'raw') # 소수점으로 생성이 되어 'class' 작동 안됨.
# wiscon_nnet_pred <- round(wiscon_nnet_pred, 0) # 소수점 정리
# table(wiscon_nnet_pred, wiscon_test$diagnosis)
# (100+60) / 171 # 93% 분류정확도 # 1% 내외로 돌릴때 마다 값이 변함.

# 1-4 랜덤포레스트 분류기법
install.packages('randomForest')
library(randomForest)

train$diagnosis[train$diagnosis == 'B'] <- 1
train$diagnosis[train$diagnosis == 'M'] <- 2
test$diagnosis[test$diagnosis == 'B'] <- 1
test$diagnosis[test$diagnosis == 'M'] <- 2
train$diagnosis <- as.factor(train$diagnosis) # 캐릭터값에서 변경
test$diagnosis <- as.factor(test$diagnosis)

str(train) # 31개 변수중 종속변수를 제외하면 30개
sqrt(30) # 5.4개이므로 변수는 5개로 사용

wiscon_rf <- randomForest(diagnosis ~ ., test, mtry = 5, importance = T) # ntree = 500  
wiscon_rf  # 랜덤포레스트 분석 결과 # 트리 갯수에 따라 OOB 수치 달라짐
(99+65) / 171 # 95%

# OOB : 4.09%

importance(wiscon_rf) # 중요 변수 확인

# 중요 변수 시각화
windows()
varImpPlot(wiscon_rf)

# 2. Abalone Data 데이터셋을 대상으로 전복의 나이를 예측하고자 한다. 예측기법 2개를 적용하여 기법별 결과를 비교하시오.
# -종속변수는 Rings를사용
rm(list = ls())
getwd()
setwd('D:/')
abalone_data <- read.csv('abalone.csv', header = T)
head(abalone_data); str(abalone_data) #데이터 확인

# 학습데이터, 검정데이터 셋으로 분리
set.seed(1) # 매번 값을 변동 안시키기 위해 설정
idx2 <- sample(1:nrow(abalone_data), 0.7*nrow(abalone_data))
aba_train <- abalone_data[idx2,]
aba_test <- abalone_data[-idx2,]
dim(aba_train); dim(aba_test)
head(aba_train); head(aba_test)
str(aba_train); str(aba_test)

# 2-1 의사결정나무 예측기법
library(rpart)
library(rpart.plot)

abalon_rpart <- rpart(Rings ~ ., data = aba_train) #의사결정트리 생성
summary(abalon_rpart) # 2923개 트리와 중요변수들 확인

windows()
rpart.plot(abalon_rpart) # 트리 시각화

# 예측 및 결과
abalon_pred <- predict(abalon_rpart, aba_test)
cor(abalon_pred, aba_test$Rings) # 63% 예측값
# 상관계수을 이용한 검정 결과 : 0.6393875

# 2-1-1 의사결정 나무 / CART 이용
# CARTTree <- rpart(Rings ~ ., data = abalone_data) # 의사결정나무 생성
# CARTTree

# windows()
# plot(CARTTree, margin = 0.2)
# text(CARTTree, cex = 1) # 의사결정나무 시각화

# CARTTree를 이용하여 abalon_data Rings 전체를 대상으로 예측
# predict(CARTTree, newdata = abalone_data, type = 'matrix')
# vector , prob, class, matrix
# 위의 결과 저장
# predicted <- predict(CARTTree, newdata = abalone_data, type = 'matrix')
# predicted <- round(predicted, 2)

# 3-5 예측정확도
# sum(predicted == abalone_data$Rings) / NROW(predicted)

# 3-6 실제값과 예측값의 비교
# real <- abalone_data$Rings
# table(real, predicted)

# 2-2 인공신경망 예측기법
# install.packages('neuralnet')
# library(neuralnet)

# aba_train$Sex[aba_train$Sex == 'M'] <- 1
# aba_train$Sex[aba_train$Sex == 'F'] <- 2
# aba_train$Sex[aba_train$Sex == 'I'] <- 3
# aba_test$Sex[aba_test$Sex == 'M'] <- 1
# aba_test$Sex[aba_test$Sex == 'F'] <- 2
# aba_test$Sex[aba_test$Sex == 'I'] <- 3
# aba_train$Sex <- as.numeric(aba_train$Sex) # 캐릭터값에서 변경
# aba_test$Sex <- as.numeric(aba_test$Sex)

# normal2 <- function(x){
#   return((x-min(x))/(max(x)-min(x)))}

# abatrain_nor <- as.data.frame(lapply(aba_train, normal2))
# abatest_nor <- as.data.frame(lapply(aba_test, normal2))

# 인공신경망 분석 실행
# abalone_neural <- neuralnet(Rings ~ Length + Diameter + Height +
#                               Whole.weight + Shucked.weight + Viscera.weight +
#                               Shell.weight + Sex, 
#                               data = abatrain_nor, hidden = 3)

# windows()
# plot(abalone_neural)  # 인공신경망 시각화

# 모델 성능평가 및 예측
# abalon_neural_result <- compute(abalone_neural, abatest_nor[1:8])
# aba_neural <- abalon_neural_result$net.result
# cor(aba_neural, abatest_nor$Rings) # 49% 정확도 매번 예측값이 다름
# 상관계수을 이용한 분류정확도 검정 결과 : 0.4943108

# 2-3 랜덤포레스트 예측기법
rm(list = ls())
library(randomForest)
getwd()
setwd('D:/')
abalone_data <- read.csv('abalone.csv', header = T)
head(abalone_data); str(abalone_data) #데이터 확인

aba_train$Sex <- as.factor(aba_train$Sex) # 캐릭터값에서 변경
aba_test$Sex <- as.factor(aba_test$Sex)
str(aba_train) # 9개 변수중 종속변수를 제외하면 8개
sqrt(8) # 2.8개이므로 변수는 3개로 사용

abalon_rf <- randomForest(Rings ~ ., data = aba_train, mtry = 3, importance = T)
windows()
plot(abalon_rf)  # 트리 개수 변화에 따른 오류 감소 추이

# 변수 중요도 
importance(abalon_rf)  # 변수 중요도 확인
windows()
varImpPlot(abalon_rf)  # 변수 중요도 플로팅

# 예측 및 결과
abalon_rf_pred <- predict(abalon_rf, newdata = aba_test)
cor(abalon_rf_pred, aba_test$Rings) # 73% 예측값.
# 상관계수을 이용한 검정 결과 : 0.730932
windows()
plot(abalon_rf_pred)
points(aba_test$Rings, col = "red")

# 2-4 선형회귀분석
library(car)
abalon_lm <- lm(Rings ~ ., data = aba_train)
summary(abalon_lm)

abalon_lm_res <- residuals(abalon_lm)
durbinWatsonTest(abalon_lm_res)
# 1.996 이므로 독립선 상한인 1.69보다 크므로 계수들은 서로 독립적이다 볼 수 있음.

windows()
par(mfrow = c(1,1))
plot(abalon_lm)  # 등분산성 확인

shapiro.test(abalon_lm_res)

# 예측 및 결과
abalon_data_pred <- predict(abalon_lm, newdata = aba_test)
cor(abalon_data_pred, aba_test$Rings) # 상관계수 결과 0.7101704 / 71%
windows()
plot(abalon_data_pred)
points(aba_test$Rings, col = "red")

# 3. iris데이터에서 species 컬럼 데이터를 제거한 후 k-means clustering를 실행하고 시각화하시오.
install.packages("mclust")
library(mclust)
data(iris)

# Species 컬럼 제거
iris$Species <- NULL
head(iris)

# k-means clustering 생성
iris_kmeans <- kmeans(iris, 6)
iris_kmeans
str(iris_kmeans)

# 시각화
windows()
par(mfrow = c(1,2))
plot(iris[c('Sepal.Length', 'Sepal.Width')], col = iris_kmeans$cluster)
points(iris_kmeans$centers[, c('Sepal.Length', 'Sepal.Width')], col=1:4, pch=8, cex=2)
plot(iris[c('Petal.Length', 'Petal.Width')], col = iris_kmeans$cluster)
points(iris_kmeans$centers[, c('Petal.Length', 'Petal.Width')], col=1:4, pch=8, cex=2)