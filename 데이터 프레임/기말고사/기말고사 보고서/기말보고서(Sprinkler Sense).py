# 라이브러리 불러오기
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# 한글 폰트 설정 ----> 초기 소스코드에서 나온 그래프에서 폰트 문제가 생겨 위와 같은 소스코드를 추가
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 절대 경로가 아닌 상대 경로 설정 및 데이터 로드하였음
file_path = "../기말데이터/1~6파일_정리_파일.csv"

try:
    final_merged_data = pd.read_csv(file_path)  # 데이터 읽기
    print("데이터 로드 완료! 데이터 크기:", final_merged_data.shape)
except FileNotFoundError:
    print(f"파일을 찾을 수 없습니다: {file_path}")
    raise

# 데이터 요약 및 탐색
print("\n데이터 요약:")
print(final_merged_data.info())  # 데이터 타입 및 결측값 확인
print("\n기본 통계 요약:")
print(final_merged_data.describe())  # 기본 통계 요약

# 데이터 분포 시각화 (Value, Temperature, Distance)
plt.figure(figsize=(8, 6))
sns.histplot(final_merged_data["Value"].dropna(), kde=True, bins=30, color="blue")
plt.title("Value 분포")
plt.xlabel("Value")
plt.ylabel("빈도")
plt.savefig("./value_distribution.png")  # 그래프 저장
plt.close()  # 메모리 해제

# 상관관계 분석 (수치형 데이터만)
numeric_data = final_merged_data.select_dtypes(include=[np.number])
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_data.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("수치형 변수 간 상관관계")
plt.savefig("./correlation_plot.png")  # 그래프 저장
plt.close()  # 메모리 해제

# 결측값 처리
final_merged_data = final_merged_data.dropna(subset=["Value"])  # 'Value' 열의 결측값 제거
final_merged_data["Temperature"] = final_merged_data["Temperature"].fillna(final_merged_data["Temperature"].mean())
final_merged_data["Distance"] = final_merged_data["Distance"].fillna(final_merged_data["Distance"].mean())

# 로그 변환 (종속 변수)
final_merged_data["Log_Value"] = np.log1p(final_merged_data["Value"])

# 입력 변수(X)와 종속 변수(y) 준비
features = ["Surface", "Temperature", "BaseTemperature", "Distance",
            "MondayIsDayOff", "TuesdayIsDayOff", "WednesdayIsDayOff",
            "ThursdayIsDayOff", "FridayIsDayOff", "SaturdayIsDayOff", "SundayIsDayOff"]
X = final_merged_data[features]
y = final_merged_data["Log_Value"]

# 학습 데이터와 테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 랜덤 포레스트 회귀 모델 학습
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 모델 평가
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)  # RMSE 계산
r2 = r2_score(y_test, y_pred)
print("\n모델 평가 결과:")
print("평균 제곱 오차 (MSE):", mse)
print("RMSE (Root Mean Squared Error):", rmse)
print("결정 계수 (R²):", r2)

# 실제값 vs 예측값 시각화
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color="blue")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title("실제값 vs 예측값 (로그 변환된 값)")
plt.xlabel("실제 로그 값")
plt.ylabel("예측 로그 값")
plt.savefig("./actual_vs_predicted.png")  # 그래프 저장
plt.close()

# 잔차 분석
residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
sns.histplot(residuals, kde=True, color="purple", bins=30)
plt.title("잔차 분포")
plt.xlabel("잔차 (실제값 - 예측값)")
plt.ylabel("빈도")
plt.savefig("./residuals_distribution.png")  # 그래프 저장
plt.close()

# 변수 중요도 시각화
feature_importances = model.feature_importances_
importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": feature_importances
}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=importance_df, palette="viridis", hue="Feature")
plt.title("입력 변수 중요도")
plt.xlabel("중요도")
plt.ylabel("입력 변수")
plt.savefig("./feature_importance.png")  # 그래프 저장
plt.close()

from sklearn.model_selection import KFold, cross_val_score

# K-Fold 교차 검증 설정
kf = KFold(n_splits=5, shuffle=True, random_state=42)  # 5-Fold 교차 검증 설정

# MSE, RMSE, R^2 점수 저장
mse_scores = []
rmse_scores = []  # RMSE 저장 리스트 추가
r2_scores = []

# K-Fold로 데이터 나누고 모델 평가
for train_index, test_index in kf.split(X):
    X_train_kf, X_test_kf = X.iloc[train_index], X.iloc[test_index]
    y_train_kf, y_test_kf = y.iloc[train_index], y.iloc[test_index]

    # 모델 학습
    model.fit(X_train_kf, y_train_kf)

    # 예측 및 평가
    y_pred_kf = model.predict(X_test_kf)
    mse_fold = mean_squared_error(y_test_kf, y_pred_kf)
    rmse_fold = np.sqrt(mse_fold)  # RMSE 계산
    mse_scores.append(mse_fold)
    rmse_scores.append(rmse_fold)  # RMSE 저장
    r2_scores.append(r2_score(y_test_kf, y_pred_kf))

# 평균 성능 출력
mean_mse = np.mean(mse_scores)
mean_rmse = np.mean(rmse_scores)  # 평균 RMSE 계산
mean_r2 = np.mean(r2_scores)
print("\nK-Fold 교차 검증 결과:")
print(f"평균 MSE: {mean_mse:.3f}")
print(f"평균 RMSE: {mean_rmse:.3f}")
print(f"평균 R^2: {mean_r2:.3f}")

# 시각화: Fold별 MSE, RMSE, R^2 점수
plt.figure(figsize=(10, 6))
plt.plot(range(1, kf.get_n_splits() + 1), mse_scores, marker='o', label='MSE', color='blue')
plt.plot(range(1, kf.get_n_splits() + 1), rmse_scores, marker='o', label='RMSE', color='red')
plt.plot(range(1, kf.get_n_splits() + 1), r2_scores, marker='o', label='R^2', color='green')
plt.title('K-Fold 교차 검증 결과 (MSE, RMSE, R²)')
plt.xlabel('Fold')
plt.ylabel('Score')
plt.legend()
plt.savefig("./kfold_results_rmse.png")  # 그래프 저장
plt.close()
