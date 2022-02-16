# 사례연구4 문태웅

#### 1)
# 위스콘신 유방암 데이터셋을 대상으로 분류기법 2개를 적용하여
# 기법별 결과를 비교하시오.
# -종속변수는diagnosis: Benign(양성), Malignancy(악성)

# 로지스틱회귀분석은 분류 분석에 속하는 분석으로 종속변수가 범주형인 경우 새로운 자료에 대한 분류 목적으로 사용된다. 

# 분석기법1) 로지스틱 희귀분석
# 참고자료) R 교재 493p 및 인터넷 사이트

# 데이터 불러오기
getwd()
setwd('D:/OneDrive/R/사례연구')
getwd()
data_1<-read.csv('wisc_bc_data.csv') # 유방암 데이터 불러오기

# 불러온 데이터 행수 열수 확인
nrow(data_1) # 행수 확인 569개
ncol(data_1) # 열수 확인 32개

# 데이터 정리
# 불필요한 식별번호 데이터 제거
data_2 <- data_1[-1]
View(data_2) # 식별번호 제거 확인

# 족송변수의 진단 결과를 0과 1로 코딩
data_2$diagnosis[data_2$diagnosis == 'B'] <- 1 # 양성은 1
data_2$diagnosis[data_2$diagnosis == 'M'] <- 0 # 악성은 0
# 데이터 변경 확인
View(data_2)

# 종속변수의 타입 변경
data_2$diagnosis <- as.numeric(data_2$diagnosis)
# 숫자형으로 변경되었는지 확인
mode(data_2$diagnosis)

set.seed(1)
# sample 함수를 사용해 학습 및 검정데이터 생성 (60:40 비율로 생성)
data_3 <- sample(1:nrow(data_2),nrow(data_2)*0.7)

real <- data_2[data_3,] # data_3 숫자에 포함된 값과 일치하는 data_4 행이 들어간 변수 생성
test <- data_2[-data_3,] # - 기호를 붙임으로써 data_3 포함되지 않은 data_4 행이 들어간 변수 생성


# 로지스틱 분석 시작
log_1 <- glm(diagnosis~., data = real)
summary(log_1)

#회귀모델 예측치 생성 과정
pred <- predict(log_1, newdata = test, type ='response')
#시그모이드 함수 : 0.5 기준 -> 양성, 악성 판단
result_pred <- ifelse(pred >= 0.5, 1,0)
result_pred

#모델 평가
table(result_pred, test$diagnosis)
# 정확도 측정 -결과값
(60+102) / nrow(test)
# 0.9473684


# 그래프(커브)를 이용한 모델 평가
install.packages("ROCR")
library(ROCR)
pr <- prediction(pred,test$diagnosis)
prf <- performance(pr, measure = "tpr",x.measure = "fpr")
X11();plot(prf)


#### 2)
# Abalone Data 데이터셋을 대상으로 전복의 나이를 예측하고자 한다. 예측기법 2개를 적용하여
# 기법별 결과를 비교하시오.
# -종속변수는 Rings를사용

getwd()
setwd('D:/OneDrive/R/사례연구')
getwd()
df <- read.csv('abalone.csv', header=T, as.is=T) # 아발론 데이터 불러오기
head(df)


set.seed(1) # 동일한 결과값을 위해시드는 팀원과 통일

# sample 함수를 사용해 학습 및 검정데이터 생성 (7:3 비율로 생성)
df_1 <- sample(1:nrow(df),nrow(df)*0.7)
real_1 <- df[df_1,] # data_3 숫자에 포함된 값과 일치하는 data_4 행이 들어간 변수 생성
test_1 <- df[-df_1,] # - 기호를 붙임으로써 data_3 포함되지 않은 data_4 행이 들어간 변수 생성
# 몇개의 데이터가 추출되었는지 확인
nrow(real_1); nrow(test_1)

# 의자결정나무를 하기 위해 rpart 실행
library(rpart)
library(rpart.plot) # 의사결정나무 시각화 패키지 실행

# rpart 패키지를 이용하여 의사 결정 나무를 만든다.
data_tr <- rpart(Rings ~ ., data = real_1)
# 만들어진 의사결정나무 확인
data_tr
# 전체 데이터 2923개

# 트리 시각화
rpart.plot(data_tr)

# 예측 및 결과
abalon_1 <- predict(data_tr, test_1)
cor(abalon_1, test_1$Rings) # 0.6845941 예측값

# 랜덤 포레스트
# 패키지 설치 및 실행
install.packages("randomForest")
library(randomForest)

# set.seed(1) # 셋 시드를 사용할 경우 랜덤 포레스트 값이 고정되어 주석처리
RFmodel <- randomForest(Rings~., data=real_1, ntree=100, proximity=T)
RFmodel
plot(RFmodel, main="RandomForest Model of abalon")

# 사용된 변수 중 중요한 것
importance(RFmodel)
# 중요한 변수 시각화
varImpPlot(RFmodel)


# 테스트 데이터로 예측
pred <- predict(RFmodel, newdata=test_1)
# 분류정확도
cor(pred, test_1$Rings)
# 0.7485003


#### 3)
# iris데이터에서 species 컬럼 데이터를
# 제거한 후 k-means clustering를 실행하고 시각화하시오.
# 남원식님 도움으로 자료 보완
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
# 남원식님 도움으로 최적의 군집수 찾기 추가
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

