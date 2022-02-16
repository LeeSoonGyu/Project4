# 사례연구4 남원식, 문태웅, 이순규

# 1. 위스콘신 유방암 데이터셋을 대상으로 분류기법 2개를 적용하여 기법별 결과를 비교하시오.
# -종속변수는diagnosis: Benign(양성), Malignancy(악성)

#### 1-1 나이브 베이즈 모형 ####
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

#### 1-2 의사결정 나무 분류기법 ####
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

# 2. Abalone Data 데이터셋을 대상으로 전복의 나이를 예측하고자 한다. 예측기법 2개를 적용하여 기법별 결과를 비교하시오.
# -종속변수는 Rings를사용

#### 2-1 선형 회귀 예측 기법 ####
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

#### 2-2 랜덤 포레스트 예측 기법 ####
rm(list = ls())
getwd()
setwd()
abalone_data <- read.csv('abalone.csv', header = T)
head(abalone_data); str(abalone_data) #데이터 확인

# 학습데이터, 검정데이터 셋으로 분리
set.seed(1) # 매번 값을 변동 안시키기 위해 설정
idx2 <- sample(1:nrow(abalone_data), 0.7*nrow(abalone_data))
aba_train <- abalone_data[idx2,]
aba_test <- abalone_data[-idx2,]

aba_train$Sex <- as.factor(aba_train$Sex) # 캐릭터값에서 변경
aba_test$Sex <- as.factor(aba_test$Sex)
str(aba_train) # 9개 변수중 종속변수를 제외하면 8개
sqrt(8) # 2.8개이므로 변수는 3개로 사용

abalon_rf <- randomForest(Rings ~ ., data = aba_train, mtry = 3, importance = T)
windows()
plot(abalon_rf)  # 트리 개수 변화에 따른 오류 감소 추이

# 변수 중요도 
importance(abalon_rf)  # 변수 중요도
windows()
varImpPlot(abalon_rf)  # 변수 중요도 시각화

# 예측 및 결과
abalon_rf_pred <- predict(abalon_rf, newdata = aba_test)
cor(abalon_rf_pred, aba_test$Rings) # 73% 예측값.
# 상관계수을 이용한 검정 결과 : 0.730932
windows()
plot(abalon_rf_pred)
points(aba_test$Rings, col = "red")

#### 3. iris데이터에서 species 컬럼 데이터를 제거한 후 k-means clustering를 실행하고 시각화하시오.####

library(ggplot2)
install.packages("mclust")
install.packages("corrgram")
install.packages("NbClust")
library(mclust)
library(corrgram)

data(iris)
head(iris)
str(iris)

iris2 <- iris

# 최적의 군집수 찾기
library(NbClust)
set.seed(1)
iris_c <- NbClust(iris2, min.nc=2, max.nc=15, method="kmeans")
table(iris_c$Best.n[1,])
# 2~ 3개의 군집수가 적당

# species 데이터 제거
iris2$Species <- NULL 
head(iris2)

# k-means clustering 실행
kmeans_result <- kmeans(iris2, 6) 
kmeans_result
str(kmeans_result)

# 시각화
plot(iris2[c("Sepal.Length", "Sepal.Width")], col=kmeans_result$cluster)
points(kmeans_result$centers[, c("Sepal.Length", "Sepal.Width")], col=1:4, pch=8, cex=2)
plot(iris2[c("Petal.Length", "Petal.Width")], col=kmeans_result$cluster)
points(kmeans_result$centers[, c("Petal.Length", "Petal.Width")], col=1:4, pch=8, cex=2)